"""浮动基座全身 IK/QP。

优化变量 dq = [基座 6 自由度, 29 个身体关节] 的单步广义速度增量：
- 双手 6-DOF 位姿：加权最小二乘任务
- 骨盆高度、躯干直立：加权任务
- 质心水平位置：加权任务（把 CoM 拉到支撑面中心）
- 双脚位姿：硬等式约束（不允许滑动）
- 支撑多边形：带松弛的不等式约束
- 姿态先验 + 阻尼正则 + 关节位置/速度限幅

基座自由度参与优化，因此解出的基座增量与“双脚固定”的闭链运动学严格一致。
"""

import time

import mujoco
import numpy as np
from qpsolvers import solve_qp
from scipy.sparse import csc_matrix

from . import config
from .utils import mat_to_quat, polygon_edges, quat_error


def _clip(vec, limit):
    norm = np.linalg.norm(vec)
    return vec * (limit / norm) if norm > limit else vec


class WholeBodyQP:
    """速度级浮动基座全身 IK。"""

    def __init__(self, robot, data=None):
        self.r = robot
        self.m = robot.model
        self.d = robot.data if data is None else data
        self.cols = robot.ik_dof_cols
        self.n = len(self.cols)
        self.dt = config.SIM_TIMESTEP
        self.jacp = np.zeros((3, self.m.nv))
        self.jacr = np.zeros((3, self.m.nv))
        self.jac_com = np.zeros((3, self.m.nv))
        self.solve_time = 0.0
        self.last_status = "uninitialized"

    # ------------------------------------------------------------------ #
    def bind(self, data):
        self.d = data

    def _site_jac(self, key):
        sid = self.r.site_ids[key]
        mujoco.mj_jacSite(self.m, self.d, self.jacp, self.jacr, sid)
        return self.jacp[:, self.cols].copy(), self.jacr[:, self.cols].copy()

    def _site_quat(self, key):
        sid = self.r.site_ids[key]
        return mat_to_quat(self.d.site_xmat[sid].reshape(3, 3))

    def _body_rot_jac(self, body_key):
        bid = self.r.body_ids[body_key]
        mujoco.mj_jacBody(self.m, self.d, None, self.jacr, bid)
        return self.jacr[:, self.cols].copy()

    # ------------------------------------------------------------------ #
    def solve(self, hand_targets, base_z, com_target, support_poly, posture_pref,
              support_lookahead=None, arm_vel_limit=None, base_vel_limit=None,
              base_angvel_limit=None, hand_vel=None, hand_acc=None,
              hand_omega=None, com_vel_ff=None, torso_quat_target=None):
        d = self.d
        dt = self.dt
        rows, rhs = [], []

        # ---- 双手 6-DOF ----
        for key in ("left_hand", "right_hand"):
            pos_t, quat_t = hand_targets[key]
            jp, jr = self._site_jac(key)
            ap = 1.0 - np.exp(-dt / config.TAU_HAND_POS)
            ar = 1.0 - np.exp(-dt / config.TAU_HAND_ROT)
            # 位置误差（限幅） + 目标速度/加速度前馈
            ep = _clip(ap * (pos_t - d.site_xpos[self.r.site_ids[key]]),
                       config.CLIP_HAND_POS)
            if hand_vel is not None:
                ep = ep + np.asarray(hand_vel[key]) * dt
            if hand_acc is not None:
                ep = ep + 0.5 * np.asarray(hand_acc[key]) * dt * dt
            # 姿态误差（限幅） + 目标角速度前馈
            er = _clip(ar * quat_error(quat_t, self._site_quat(key)),
                       config.CLIP_HAND_ROT)
            if hand_omega is not None:
                er = er + np.asarray(hand_omega[key]) * dt
            rows += [np.sqrt(config.W_HAND_POS) * jp,
                     np.sqrt(config.W_HAND_ROT) * jr]
            rhs += [np.sqrt(config.W_HAND_POS) * ep,
                    np.sqrt(config.W_HAND_ROT) * er]

        # ---- 骨盆高度 ----
        az = 1.0 - np.exp(-dt / config.TAU_BASE_Z)
        jz = np.zeros((1, self.n))
        jz[0, 2] = 1.0
        rows.append(np.sqrt(config.W_BASE_Z) * jz)
        rhs.append(np.sqrt(config.W_BASE_Z)
                   * np.array([az * (base_z - d.qpos[2])]))

        # ---- 躯干姿态先验 ----
        jr_torso = self._body_rot_jac("torso")
        torso_now = d.xquat[self.r.body_ids["torso"]]
        at = 1.0 - np.exp(-dt / config.TAU_TORSO)
        torso_target = (self.r.torso_quat if torso_quat_target is None
                        else torso_quat_target)
        rows.append(np.sqrt(config.W_TORSO) * jr_torso)
        rhs.append(np.sqrt(config.W_TORSO)
                   * _clip(at * quat_error(torso_target, torso_now), 0.15))

        # ---- 质心水平位置 ----
        mujoco.mj_jacSubtreeCom(self.m, d, self.jac_com, 0)
        com = d.subtree_com[0]
        com_vel = self.jac_com @ d.qvel
        ac = 1.0 - np.exp(-dt / config.TAU_COM)
        # 倒立摆本身无阻尼：com_vel_ff 可用于注入阻尼/前馈速度
        ec = _clip(ac * (com_target - com[:2]) - 0.5 * dt * com_vel[:2],
                   config.CLIP_COM)
        if com_vel_ff is not None:
            ec = ec + np.asarray(com_vel_ff) * dt
        rows.append(np.sqrt(config.W_COM) * self.jac_com[:2, self.cols])
        rhs.append(np.sqrt(config.W_COM) * ec)

        # ---- 姿态先验 ----
        q_now = d.qpos[self.r.body_qpos_adr]
        ap = 1.0 - np.exp(-dt / config.TAU_POSTURE)
        j_post = np.zeros((self.r.n_joints, self.n))
        j_post[:, 6:] = np.eye(self.r.n_joints)
        rows.append(np.sqrt(config.W_POSTURE) * j_post)
        rhs.append(np.sqrt(config.W_POSTURE) * ap * (posture_pref - q_now))

        # ---- 阻尼 ----
        rows.append(np.sqrt(config.W_DAMPING) * np.eye(self.n))
        rhs.append(np.zeros(self.n))

        A = np.vstack(rows)
        b = np.concatenate(rhs)
        P = A.T @ A + 1e-8 * np.eye(self.n)
        q = -A.T @ b

        # ---- 双脚硬等式：J_foot dq = 0 ----
        eq_rows = []
        for key in ("left_foot", "right_foot"):
            jp, jr = self._site_jac(key)
            eq_rows += [jp, jr]
        A_eq = np.vstack(eq_rows)
        b_eq = np.zeros(A_eq.shape[0])

        # ---- 支撑多边形不等式：n·CoM <= offset ----
        normals, offsets = polygon_edges(support_poly)
        radius = float(self.m.geom_size[self.r.foot_contact_geoms[0], 0])
        offsets = offsets - radius - config.SUPPORT_SHRINK
        com_eff = com[:2] + (config.SUPPORT_VEL_LOOKAHEAD
                             if support_lookahead is None else support_lookahead) * com_vel[:2]
        A_sup = normals @ self.jac_com[:2, self.cols]
        u_sup = offsets - normals @ com_eff
        n_sup = A_sup.shape[0]

        ns = config.SUPPORT_SLACK_N
        if n_sup > ns:
            raise RuntimeError(f"支撑边数 {n_sup} 超过松弛变量数 {ns}")
        sl = np.zeros((n_sup, ns))
        sl[:, :n_sup] = -np.eye(n_sup)
        A_sup_aug = np.hstack([A_sup, sl])

        arm_lim = config.MAX_ARM_VEL if arm_vel_limit is None else arm_vel_limit
        base_lim = config.MAX_BASE_VEL if base_vel_limit is None else base_vel_limit
        base_w_lim = (config.MAX_BASE_ANGVEL if base_angvel_limit is None
                      else base_angvel_limit)
        nj = self.r.n_joints
        vel_lim = np.full(nj, config.MAX_JOINT_VEL * dt)
        vel_lim[self.r.n_nonarm:] = arm_lim * dt
        lim_margin = self.r.joint_limit_margin
        lower = np.concatenate([
            -np.full(3, base_lim * dt),
            -np.full(3, base_w_lim * dt),
            np.maximum(self.r.body_joint_lo + lim_margin - q_now,
                       -vel_lim),
        ])
        upper = np.concatenate([
            np.full(3, base_lim * dt),
            np.full(3, base_w_lim * dt),
            np.minimum(self.r.body_joint_hi - lim_margin - q_now,
                       vel_lim),
        ])
        upper = np.maximum(upper, lower)

        ntot = self.n + ns
        P_all = np.zeros((ntot, ntot))
        P_all[:self.n, :self.n] = P
        P_all[self.n:, self.n:] = config.SUPPORT_SLACK_WEIGHT * np.eye(ns)
        q_all = np.zeros(ntot)
        q_all[:self.n] = q

        A_eq_aug = np.hstack([A_eq, np.zeros((A_eq.shape[0], ns))])
        lb = np.concatenate([lower, np.zeros(ns)])
        ub = np.concatenate([upper, np.full(ns, np.inf)])

        t0 = time.perf_counter()
        x = solve_qp(csc_matrix(P_all), q_all, G=csc_matrix(A_sup_aug), h=u_sup,
                     A=csc_matrix(A_eq_aug), b=b_eq, lb=lb, ub=ub,
                     solver="clarabel")
        self.solve_time = time.perf_counter() - t0
        if x is None:
            self.last_status = "failed"
            return np.zeros(self.n)
        self.last_status = "solved"
        return x[:self.n]
