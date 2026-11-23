"""动力学级浮动基座全身控制（WBC）。

与 `wbc.py` 的速度级运动学 IK 不同，这里直接求解加速度与力矩：

    x = [q̈(35), f_c(12), τ(29), s(12)]

约束：

- 浮动基座动力学等式：M q̈ + h = Sᵀ τ + J_cᵀ f_c
- 双脚不滑动（加速度级）：J_c q̈ + J̇_c q̇ = 0，带松弛 s
- 摩擦锥、法向力单边、CoP 在脚掌支撑多边形内、偏航力矩上限
- 关节力矩、关节/基座加速度限幅

任务（加权最小二乘）：

- 双手 6-DOF 加速度（位置/姿态 PD + 参考速度/加速度前馈）
- CoM 水平加速度（倒立摆 PD，把 CoM 拉回支撑中心）
- 骨盆高度、躯干直立
- 腿/腰姿态先验、关节阻尼

力矩通过位置执行器反解下发：
    τ = kp (ctrl - q) - kv q̇  ⇒  ctrl = q + (τ + kv q̇) / kp
"""

from __future__ import annotations

import time

import mujoco
import numpy as np
from qpsolvers import solve_qp

from . import config
from .utils import mat_to_quat, polygon_edges, quat_error


def _clip(vec, limit):
    vec = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(vec))
    return vec * (limit / norm) if norm > limit else vec


class WholeBodyWBC:
    """接触力 + 全身逆动力学的浮动基座 WBC。"""

    def __init__(self, robot, data=None):
        self.r = robot
        self.m = robot.model
        self.d = robot.data if data is None else data
        self.cols = np.asarray(robot.ik_dof_cols, dtype=int)
        self.n = len(self.cols)
        self.nj = robot.n_joints
        self.nv = self.m.nv
        self.dt = config.SIM_TIMESTEP

        self.total_mass = float(self.m.body_subtreemass[0])
        self.gravity = abs(float(self.m.opt.gravity[2]))

        self.site_body = {
            key: int(self.m.site_bodyid[sid]) for key, sid in robot.site_ids.items()
        }
        self.foot_keys = ("left_foot", "right_foot")
        self.foot_sites = {k: robot.site_ids[k] for k in self.foot_keys}
        self.foot_bodies = {
            "left_foot": robot.body_ids["left_ankle"],
            "right_foot": robot.body_ids["right_ankle"],
        }
        self.foot_geoms = self._split_foot_geoms()
        # 逐接触球建模：每个脚 4 个球，共 8 个 3D 点力
        self.contact_gids = [gid for key in self.foot_keys
                             for gid in self.foot_geoms[key]]
        self.contact_foot = {}
        self.contact_body = {}
        for key in self.foot_keys:
            for gid in self.foot_geoms[key]:
                self.contact_foot[gid] = key
                self.contact_body[gid] = int(self.m.geom_bodyid[gid])
        self.n_contacts = len(self.contact_gids)
        self.head_body = robot.body_ids["torso"]
        self.head_geom = robot.head_geom
        self.cop_bounds = self._cop_bounds()

        # 执行器参数
        self.act_ids = np.asarray(robot.body_act_ids, dtype=int)
        self.kp = self.m.actuator_gainprm[self.act_ids, 0].copy()
        self.kv = -self.m.actuator_biasprm[self.act_ids, 2].copy()
        self.tau_lim = np.abs(self.m.actuator_forcerange[self.act_ids, 0]).copy()
        self.tau_lim = np.where(self.tau_lim > 1e-6,
                                self.tau_lim * config.WBC_TAU_LIMIT_SCALE,
                                config.WBC_TAU_FALLBACK)
        self.qpos_adr = np.asarray(robot.body_qpos_adr, dtype=int)
        self.dof_adr = np.asarray(robot.body_dof_adr, dtype=int)
        self.n_nonarm = robot.n_nonarm

        # q̈ 限幅
        qdd = np.full(self.nj, config.WBC_QDD_ARM_MAX)
        qdd[:self.n_nonarm] = config.WBC_QDD_LEG_MAX
        self.qdd_joint_max = qdd

        # 缓存
        self.M = np.zeros((self.nv, self.nv))
        self.Jcom = np.zeros((3, self.nv))
        self.jacp = np.zeros((3, self.nv))
        self.jacr = np.zeros((3, self.nv))
        self.jacp_dot = np.zeros((3, self.nv))
        self.jacr_dot = np.zeros((3, self.nv))

        self.com_z0 = float(robot.data.subtree_com[0][2])
        self.base_z0 = float(robot.data.qpos[2])

        self.solve_time = 0.0
        self.last_status = "uninitialized"
        self.last_info: dict = {}

    # ------------------------------------------------------------------ #
    def _split_foot_geoms(self):
        out = {"left_foot": [], "right_foot": []}
        for gid in self.r.foot_contact_geoms:
            body = int(self.m.geom_bodyid[gid])
            if body == self.foot_bodies["left_foot"]:
                out["left_foot"].append(int(gid))
            elif body == self.foot_bodies["right_foot"]:
                out["right_foot"].append(int(gid))
        return out

    def _cop_bounds(self):
        """支撑脚接触球的 CoP 允许范围（初始站姿下的球心包围盒）。"""
        bounds = {}
        for key in self.foot_keys:
            pts = np.array([self.d.geom_xpos[g][:2] for g in self.foot_geoms[key]])
            bounds[key] = (pts.min(axis=0), pts.max(axis=0))
        return bounds

    def _site6(self, key):
        sid = self.r.site_ids[key]
        body = self.site_body[key]
        point = self.d.site_xpos[sid]
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, sid)
        mujoco.mj_jacDot(self.m, self.d, self.jacp_dot, self.jacr_dot,
                         point, body)
        return (self.jacp[:, self.cols].copy(),
                self.jacr[:, self.cols].copy(),
                self.jacp_dot[:, self.cols].copy(),
                self.jacr_dot[:, self.cols].copy())

    def _body_rot_jac(self, body_key):
        bid = self.r.body_ids[body_key]
        mujoco.mj_jacBody(self.m, self.d, None, self.jacr, bid)
        mujoco.mj_jacDot(self.m, self.d, None, self.jacr_dot,
                         self.d.xipos[bid], bid)
        return self.jacr[:, self.cols].copy(), self.jacr_dot[:, self.cols].copy()

    def _com_jac(self):
        mujoco.mj_jacSubtreeCom(self.m, self.d, self.Jcom, 0)
        return self.Jcom[:, self.cols].copy()

    # ------------------------------------------------------------------ #
    def _contact_state(self):
        """返回每个接触球是否着地，以及每只脚的接触点数。"""
        active = set()
        for c in self.d.contact:
            for g in (int(c.geom1), int(c.geom2)):
                if g in self.contact_foot:
                    active.add(g)
        per_foot = {k: 0 for k in self.foot_keys}
        for gid in active:
            per_foot[self.contact_foot[gid]] += 1
        return dict(active=active, per_foot=per_foot,
                    any=bool(active))

    # ------------------------------------------------------------------ #
    def solve(self, hands, com_ref, base_z_ref, posture, torso_ref=None,
              head_ref=None, head_gate=1.0, base_quat_ref=None,
              base_xy_ref=None):
        """求解一步 WBC。

        hands: {key: dict(pos, vel, acc, quat, omega, alpha)}
        com_ref: (2,) CoM 水平参考
        base_z_ref: 骨盆高度参考
        posture: 29 维姿态先验
        head_ref: (2,) 头部水平参考（前后/侧向），None 表示不约束
        head_gate: 0–1，头部任务权重缩放（大幅躯干前倾时降低）
        base_quat_ref: (4,) 基座姿态参考，None 表示不约束
        base_xy_ref: (2,) 骨盆水平位置参考，None 表示不约束
        """
        d = self.d
        m = self.m
        qvel = d.qvel
        qvel_c = qvel[self.cols]
        cols = self.cols
        n = self.n
        nj = self.nj
        nf = 3 * self.n_contacts
        n_tau = nj
        n_slack = 3 * self.n_contacts
        i_f = n
        i_tau = i_f + nf
        i_s = i_tau + n_tau
        nx = i_s + n_slack
        nq = nx

        # ---- 质量矩阵 / 偏置力 ----
        mujoco.mj_fullM(m, self.M, d.qM)
        Mb = self.M[np.ix_(cols, cols)]
        h = d.qfrc_bias[cols].copy()

        # ---- 逐接触点 ----
        contact = self._contact_state()
        Jc = np.zeros((nf, n))
        Jdc_qvel = np.zeros(nf)
        point_pos = np.zeros((self.n_contacts, 3))
        jp = np.zeros((3, self.nv))
        jp_dot = np.zeros((3, self.nv))
        for j, gid in enumerate(self.contact_gids):
            point = d.geom_xpos[gid]
            point_pos[j] = point
            body = self.contact_body[gid]
            mujoco.mj_jac(m, d, jp, None, point, body)
            mujoco.mj_jacDot(m, d, jp_dot, None, point, body)
            Jc[3 * j:3 * j + 3] = jp[:, cols]
            Jdc_qvel[3 * j:3 * j + 3] = jp_dot[:, cols] @ qvel_c
            # 支撑脚所有接触点始终保留（见 config.WBC_STANCE_ALL_POINTS）：
            # 不再按瞬时 ncon 把力变量清零，否则脚跟离地后 CoP 永远回不去。
            if not config.WBC_STANCE_ALL_POINTS and gid not in contact["active"]:
                Jc[3 * j:3 * j + 3] = 0.0
                Jdc_qvel[3 * j:3 * j + 3] = 0.0

        # ---- 任务堆叠 ----
        P = np.zeros((nq, nq))
        qv = np.zeros(nq)

        def add_task(A, b, w):
            A = np.asarray(A, dtype=float)
            b = np.asarray(b, dtype=float)
            if A.shape[1] == n:
                A_full = np.zeros((A.shape[0], nq))
                A_full[:, :n] = A
                A = A_full
            P[:] += w * (A.T @ A)
            qv[:] -= w * (A.T @ b)

        # 双手
        for key in ("left_hand", "right_hand"):
            sid = self.r.site_ids[key]
            Jp, Jr, Jdp, Jdr = self._site6(key)
            ref = hands[key]
            p_now = d.site_xpos[sid]
            v_now = Jp @ qvel_c
            ep = _clip(np.asarray(ref["pos"]) - p_now, config.WBC_CLIP_HAND_POS)
            ev = np.asarray(ref["vel"]) - v_now
            a_des = (np.asarray(ref["acc"]) + config.WBC_KP_HAND_POS * ep
                     + config.WBC_KD_HAND_POS * ev)
            a_des = _clip(a_des, config.WBC_A_HAND_POS_MAX)
            jdp_term = Jdp @ qvel_c if config.WBC_USE_HAND_JDDOT else np.zeros(3)
            add_task(Jp, a_des - jdp_term, config.WBC_W_HAND_POS)

            q_now = mat_to_quat(d.site_xmat[sid].reshape(3, 3))
            w_now = Jr @ qvel_c
            er = _clip(quat_error(ref["quat"], q_now), config.WBC_CLIP_HAND_ROT)
            alpha = (np.asarray(ref["alpha"])
                     + config.WBC_KP_HAND_ROT * er
                     + config.WBC_KD_HAND_ROT * (np.asarray(ref["omega"]) - w_now))
            alpha = _clip(alpha, config.WBC_A_HAND_ROT_MAX)
            jdr_term = Jdr @ qvel_c if config.WBC_USE_HAND_JDDOT else np.zeros(3)
            add_task(Jr, alpha - jdr_term, config.WBC_W_HAND_ROT)
        # CoM 水平（倒立摆 PD）。Jdot·q̇ 用差分估计在 500 Hz 下噪声很大，
        # 会破坏闭环稳定性；这里只保留位置/速度反馈，实际接触力由 QP 分配。
        Jcom = self._com_jac()
        com_now = d.subtree_com[0]
        vcom = Jcom @ qvel_c
        a_com = (config.WBC_OMEGA_LIP ** 2 * (np.asarray(com_ref) - com_now[:2])
                 - 2.0 * config.WBC_ZETA_LIP * config.WBC_OMEGA_LIP * vcom[:2])
        a_com = _clip(a_com, config.WBC_A_COM_MAX)
        add_task(Jcom[:2], a_com, config.WBC_W_COM)

        # CoM 垂直：跟随参考高度，抑制“用竖直力把身体顶起来”的跳起
        com_z_ref = self.com_z0 + (float(base_z_ref) - self.base_z0)
        a_com_z = (config.WBC_KP_COM_Z * (com_z_ref - com_now[2])
                   - config.WBC_KD_COM_Z * vcom[2])
        a_com_z = float(np.clip(a_com_z, -config.WBC_A_COM_Z_MAX,
                                config.WBC_A_COM_Z_MAX))
        add_task(Jcom[2:3], np.array([a_com_z]), config.WBC_W_COM_Z)

        # 骨盆高度
        jz = np.zeros((1, n))
        jz[0, 2] = 1.0
        a_z = (config.WBC_KP_BASE_Z * (float(base_z_ref) - d.qpos[2])
               - config.WBC_KD_BASE_Z * d.qvel[2])
        add_task(jz, np.array([a_z]), config.WBC_W_BASE_Z)

        # 骨盆水平位置：抑制“用屈膝把骨盆整体平移出去”的伪策略
        if base_xy_ref is not None:
            A_xy = np.zeros((2, n))
            A_xy[0, 0] = 1.0
            A_xy[1, 1] = 1.0
            a_xy = (config.WBC_KP_BASE_XY * (np.asarray(base_xy_ref) - d.qpos[:2])
                    - config.WBC_KD_BASE_XY * d.qvel[:2])
            a_xy = _clip(a_xy, config.WBC_A_BASE_XY_MAX)
            add_task(A_xy, a_xy, config.WBC_W_BASE_XY)

        # 脚掌保持水平（允许绕踝关节转动，不再用硬角加速度约束锁死脚姿态）
        for key in self.foot_keys:
            if contact["per_foot"][key] == 0:
                continue
            sid = self.r.site_ids[key]
            Jp_f, Jr_f, Jdp_f, Jdr_f = self._site6(key)
            # 脚 site 位置反馈，抑制扭矩控制下 8–9 mm 的缓慢滑移
            p_now = d.site_xpos[sid]
            v_now = Jp_f @ qvel_c
            p_ref = self.r.foot_anchor[key][0]
            a_p = (config.WBC_KP_FOOT_POS * (p_ref - p_now)
                   - config.WBC_KD_FOOT_POS * v_now)
            a_p = _clip(a_p, config.WBC_A_FOOT_POS_MAX)
            add_task(Jp_f, a_p - Jdp_f @ qvel_c, config.WBC_W_FOOT_POS)

            q_now = mat_to_quat(d.site_xmat[sid].reshape(3, 3))
            w_now = Jr_f @ qvel_c
            ref_q = self.r.foot_anchor[key][1]
            alpha_f = (config.WBC_KP_FOOT_ORIENT * quat_error(ref_q, q_now)
                       - config.WBC_KD_FOOT_ORIENT * w_now)
            alpha_f = _clip(alpha_f, config.WBC_A_FOOT_ORIENT_MAX)
            add_task(Jr_f, alpha_f - Jdr_f @ qvel_c,
                     config.WBC_W_FOOT_ORIENT)

        # 躯干直立
        Jr_t, Jdr_t = self._body_rot_jac("torso")
        torso_now = d.xquat[self.r.body_ids["torso"]]
        ref_t = self.r.torso_quat if torso_ref is None else torso_ref
        w_t = Jr_t @ qvel_c
        alpha_t = (config.WBC_KP_TORSO * quat_error(ref_t, torso_now)
                   - config.WBC_KD_TORSO * w_t)
        a_t = _clip(alpha_t, config.WBC_A_TORSO_MAX)
        add_task(Jr_t, a_t - Jdr_t @ qvel_c, config.WBC_W_TORSO)

        # 头部水平稳定：G1 没有颈关节，头是 torso 上的固定点。任务直接约束
        # 头部点前后/侧向加速度，让 WBC 优先用踝/髋平移而不是躯干俯仰来平衡，
        # 这正是“手操作时头部不要前后晃”的物理落点。
        if head_ref is not None and head_gate > 0.0:
            p_head = d.geom_xpos[self.head_geom]
            jp_h = np.zeros((3, self.nv))
            jp_h_dot = np.zeros((3, self.nv))
            mujoco.mj_jac(m, d, jp_h, None, p_head, self.head_body)
            mujoco.mj_jacDot(m, d, jp_h_dot, None, p_head, self.head_body)
            Jh = jp_h[:, cols]
            Jhd = jp_h_dot[:, cols] @ qvel_c
            v_head = Jh @ qvel_c
            a_head = (config.WBC_KP_HEAD_POS * (np.asarray(head_ref) - p_head[:2])
                      - config.WBC_KD_HEAD_POS * v_head[:2])
            a_head = _clip(a_head, config.WBC_A_HEAD_MAX)
            add_task(Jh[0:1], np.array([a_head[0] - Jhd[0]]),
                     config.WBC_W_HEAD_POS * head_gate)
            add_task(Jh[1:2], np.array([a_head[1] - Jhd[1]]),
                     config.WBC_W_HEAD_POS * config.WBC_HEAD_Y_SCALE * head_gate)

        # 基座姿态：浮基自由度本身很轻（相对腿/腰），加一个显式直立任务
        # 可以防止手臂反作用先推动基座、再经躯干任务间接拉回。
        if base_quat_ref is not None:
            Jr_b = np.zeros((3, n))
            Jr_b[:, 3:6] = np.eye(3)
            w_b = qvel_c[3:6]
            alpha_b = (config.WBC_KP_BASE_ORIENT
                       * quat_error(np.asarray(base_quat_ref), d.qpos[3:7])
                       - config.WBC_KD_BASE_ORIENT * w_b)
            alpha_b = _clip(alpha_b, config.WBC_A_BASE_ORIENT_MAX)
            add_task(Jr_b, alpha_b, config.WBC_W_BASE_ORIENT)

        # 姿态先验 + 关节阻尼
        qj = d.qpos[self.qpos_adr]
        qdj = d.qvel[self.dof_adr]
        A_j = np.zeros((nj, n))
        A_j[:, 6:] = np.eye(nj)
        qdd_post = (config.WBC_KP_POSTURE * (posture - qj)
                    - config.WBC_KD_POSTURE * qdj)
        add_task(A_j[:self.n_nonarm], qdd_post[:self.n_nonarm],
                 config.WBC_W_POSTURE)
        qdd_arm = (config.WBC_KP_POSTURE_ARM * (posture[self.n_nonarm:] - qj[self.n_nonarm:])
                   - config.WBC_KD_POSTURE_ARM * qdj[self.n_nonarm:])
        add_task(A_j[self.n_nonarm:], qdd_arm, config.WBC_W_POSTURE_ARM)
        add_task(A_j, -config.WBC_KD_JOINT_DAMP * qdj, config.WBC_W_DAMP)

        # ---- 正则项 ----
        P[:n, :n] += config.WBC_W_QDD_REG * np.eye(n)
        P[i_tau:i_tau + n_tau, i_tau:i_tau + n_tau] += (
            config.WBC_W_TAU_REG * np.eye(n_tau))
        force_nom = np.zeros(nf)
        for j in range(self.n_contacts):
            force_nom[3 * j + 2] = self.total_mass * self.gravity / self.n_contacts
        P[i_f:i_f + nf, i_f:i_f + nf] += config.WBC_W_FORCE_REG * np.eye(nf)
        qv[i_f:i_f + nf] -= config.WBC_W_FORCE_REG * force_nom
        P[i_s:i_s + n_slack, i_s:i_s + n_slack] += (
            config.WBC_W_SLACK * np.eye(n_slack))

        # ---- 等式：动力学 + 接触不滑动 ----
        S = np.zeros((n, n_tau))
        S[6:, :] = np.eye(n_tau)
        A_dyn = np.zeros((n, nq))
        A_dyn[:, :n] = Mb
        A_dyn[:, i_f:i_f + nf] = -Jc.T
        A_dyn[:, i_tau:i_tau + n_tau] = -S
        b_dyn = -h

        eq_rows, eq_rhs = [A_dyn], [b_dyn]
        row_s = 0
        for j, gid in enumerate(self.contact_gids):
            if not config.WBC_STANCE_ALL_POINTS and gid not in contact["active"]:
                continue
            A_ct = np.zeros((3, nq))
            A_ct[:, :n] = Jc[3 * j:3 * j + 3]
            A_ct[:, i_s + row_s:i_s + row_s + 3] = np.eye(3)
            eq_rows.append(A_ct)
            eq_rhs.append(-Jdc_qvel[3 * j:3 * j + 3])
            row_s += 3
        A_eq = np.vstack(eq_rows)
        b_eq = np.concatenate(eq_rhs)

        # ---- 不等式：每个接触点的摩擦锥 + 总法向力 ----
        G_rows, h_rows = [], []
        mu = config.WBC_MU_FOOT
        for j, gid in enumerate(self.contact_gids):
            if not config.WBC_STANCE_ALL_POINTS and gid not in contact["active"]:
                continue
            base = i_f + 3 * j
            for coef in ((1.0, 0.0, -mu), (-1.0, 0.0, -mu),
                         (0.0, 1.0, -mu), (0.0, -1.0, -mu)):
                row = np.zeros(nq)
                row[base:base + 3] = coef
                G_rows.append(row)
                h_rows.append(0.0)

        fz_row = np.zeros(nq)
        for j, gid in enumerate(self.contact_gids):
            if config.WBC_STANCE_ALL_POINTS or gid in contact["active"]:
                fz_row[i_f + 3 * j + 2] = 1.0
        if contact["any"]:
            mg = self.total_mass * self.gravity
            G_rows.append(fz_row.copy())
            h_rows.append(config.WBC_FZ_TOTAL_MAX_SCALE * mg)
            G_rows.append(-fz_row)
            h_rows.append(-config.WBC_FZ_TOTAL_MIN_SCALE * mg)

        # CoP 必须留在每只脚的接触点包围盒内（线性化：Σ(x_i-x_max)fz_i ≤ 0）。
        if config.WBC_USE_COP:
            shrink = config.WBC_COP_SHRINK
            for key in self.foot_keys:
                idx = [j for j, gid in enumerate(self.contact_gids)
                       if self.contact_foot[gid] == key]
                if not idx:
                    continue
                xyz = np.array([d.geom_xpos[self.contact_gids[j]] for j in idx])
                x_lo, y_lo = xyz[:, :2].min(axis=0) + shrink
                x_hi, y_hi = xyz[:, :2].max(axis=0) - shrink
                for axis, lo, hi in ((0, x_lo, x_hi), (1, y_lo, y_hi)):
                    row_hi = np.zeros(nq)
                    row_lo = np.zeros(nq)
                    for j, p in zip(idx, xyz):
                        row_hi[i_f + 3 * j + 2] = p[axis] - hi
                        row_lo[i_f + 3 * j + 2] = lo - p[axis]
                    G_rows.append(row_hi)
                    h_rows.append(0.0)
                    G_rows.append(row_lo)
                    h_rows.append(0.0)

        G = np.vstack(G_rows) if G_rows else None
        h_ineq = np.asarray(h_rows) if h_rows else None

        # ---- 盒约束 ----
        lb = np.full(nq, -np.inf)
        ub = np.full(nq, np.inf)
        lb[0:3] = -config.WBC_QDD_BASE_LIN_MAX
        ub[0:3] = config.WBC_QDD_BASE_LIN_MAX
        lb[3:6] = -config.WBC_QDD_BASE_ANG_MAX
        ub[3:6] = config.WBC_QDD_BASE_ANG_MAX
        lb[6:n] = -self.qdd_joint_max
        ub[6:n] = self.qdd_joint_max

        for j, gid in enumerate(self.contact_gids):
            base = i_f + 3 * j
            if not config.WBC_STANCE_ALL_POINTS and gid not in contact["active"]:
                lb[base:base + 3] = 0.0
                ub[base:base + 3] = 0.0
                continue
            lb[base + 0:base + 2] = -config.WBC_FXY_MAX
            ub[base + 0:base + 2] = config.WBC_FXY_MAX
            lb[base + 2] = config.WBC_FZ_POINT_MIN
            ub[base + 2] = config.WBC_FZ_POINT_MAX

        lb[i_tau:i_tau + n_tau] = -self.tau_lim
        ub[i_tau:i_tau + n_tau] = self.tau_lim
        lb[i_s:i_s + n_slack] = -config.WBC_SLACK_MAX
        ub[i_s:i_s + n_slack] = config.WBC_SLACK_MAX

        # ---- 求解 ----
        t0 = time.perf_counter()
        x = solve_qp(P, qv, G=G, h=h_ineq, A=A_eq, b=b_eq,
                     lb=lb, ub=ub, solver="clarabel")
        self.solve_time = time.perf_counter() - t0

        if x is None:
            # 放宽 CoP/偏航约束再试一次，仍失败则交给上层回退
            x = solve_qp(P, qv, G=G, h=h_ineq, A=A_eq, b=b_eq,
                         lb=lb, ub=ub, solver="osqp")
            self.solve_time = time.perf_counter() - t0

        if x is None:
            self.last_status = "failed"
            self.last_info = dict(contact=contact)
            return None

        self.last_status = "solved"
        self.last_info = dict(
            contact=contact,
            qdd=x[:n],
            force=x[i_f:i_f + nf],
            tau=x[i_tau:i_tau + n_tau],
            slack=x[i_s:i_s + n_slack],
            com=np.asarray(com_ref, float).copy(),
            a_com=np.asarray(a_com, float).copy(),
            Jcom=Jcom.copy(),
            vcom=vcom.copy(),
            solve_time=self.solve_time,
        )
        return x[i_tau:i_tau + n_tau].copy()

    # ------------------------------------------------------------------ #
    def apply(self, tau):
        """把关节力矩反解为位置执行器指令。"""
        if tau is None:
            return
        d = self.d
        q = d.qpos[self.qpos_adr]
        qd = d.qvel[self.dof_adr]
        # 关节力矩阻尼：抑制接触/位置执行器之间的高频极限环
        tau = np.asarray(tau, float) - config.WBC_TAU_DAMP * qd
        ctrl = q + (tau + self.kv * qd) / np.maximum(self.kp, 1e-6)
        ctrl = np.clip(ctrl, self.r.body_joint_lo, self.r.body_joint_hi)
        d.ctrl[self.act_ids] = ctrl
        d.ctrl[self.r.finger_act_ids] = self.r.finger_ctrl


def _convex_hull(points):
    """二维凸包（单调链），返回逆时针顶点。"""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])
