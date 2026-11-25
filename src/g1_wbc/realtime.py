"""实时闭环全身 IK：虚拟参考状态 + 关节位置伺服。

与 `sim.py` 的区别：

- `sim.plan()` 先开环积分出整条参考轨迹，`sim.replay()` 再回放；
- 这里没有预规划，每一步都在线求解，因此手部目标可以在运行中实时改变。

关键实现约束（此前版本站不住的原因）：

1. **不能只取关节部分**。`WholeBodyQP` 的解是 `dq = [基座 6, 关节 29]`，基座和
   关节必须一起积分进一个虚拟参考状态 `dv`，否则脚部等式和 CoM/高度任务会失效。
2. **参考状态必须运动学一致**。`dv` 从真实状态初始化后只做全量积分，不在每步把
   基座重新锚到真实状态，否则虚拟脚会漂移，QP 会在错误的位置上维持脚部约束。
3. **只下发关节角**。真实基座由接触自然跟随；位置伺服跟踪 `dv` 的关节角即可。

两种目标源：

- `trajectory_target(pipe)`：内置场景轨迹（按 `pre_hold` 正确对齐）；
- `mocap_target(pipe)`：读取两个 mocap 标记作为目标，可在窗口中拖动。

目标层（Step 1A）：

- 二阶临界阻尼参考，输出位置/速度/加速度，作为手部任务前馈；
- mocap 拖动时 CoM 反向配重（手向前，重心后坐），并限制在支撑多边形内；
- 预测守卫用目标加速度估算所需 ZMP 偏移，提前减速；
- 平衡工作空间限幅：目标超出固定双脚可达范围时停在边界，而不是把机器人拽倒。
"""

import time

import mujoco
import numpy as np

from . import config
from .utils import (polygon_edges, quat_error, quat_mul, quat_slerp,
                    torso_pitch_deg)
from .trajectory import quat_from_rpy
from .wbc import WholeBodyQP
from .wbc_dyn import WholeBodyWBC

_VIEWER_SYNC_DT = 0.01          # 约 100 Hz 刷新窗口


# ---------------------------------------------------------------------- #
# 目标源
# ---------------------------------------------------------------------- #
def trajectory_target(pipe):
    """目标来自场景轨迹（与预规划模式使用同一个时间轴）。"""

    def target(step, _t):
        return pipe.traj.hand(pipe.t_traj(step))

    return target


def trajectory_base_z(pipe):
    """骨盆高度参考，与场景轨迹同步。"""

    def base_z(step, _t):
        return pipe.traj.base_z(pipe.t_traj(step))

    return base_z


def standing_base_z(pipe):
    """固定站立高度，用于 mocap 交互模式。"""
    height = float(pipe.traj.base_z(0.0))

    def base_z(_step, _t):
        return height

    return base_z


def init_markers(pipe):
    """把两个 mocap 标记放到当前手部位姿，作为可拖动目标的起点。"""
    r, d = pipe.robot, pipe.robot.data
    for key in ("left_hand", "right_hand"):
        mid = pipe.mocap_ids[key]
        d.mocap_pos[mid] = d.site_xpos[r.site_ids[key]]
        d.mocap_quat[mid] = r.site_quat(key)


def mocap_target(pipe):
    """目标直接取 mocap 标记当前位姿（拖动标记 = 改目标）。"""
    d = pipe.robot.data

    def target(_step, _t):
        return {
            key: (d.mocap_pos[pipe.mocap_ids[key]].copy(),
                  d.mocap_quat[pipe.mocap_ids[key]].copy())
            for key in ("left_hand", "right_hand")
        }

    return target


# ---------------------------------------------------------------------- #
# 工具
# ---------------------------------------------------------------------- #
def _empty_log():
    return {k: [] for k in (
        "t", "hand_pos_t", "hand_quat_t", "hand_pos", "hand_quat",
        "foot_pos", "foot_quat", "base_pos", "qpos", "qvel", "ctrl",
        "q_ref", "com", "ncon", "margin", "governor", "solve_time", "status",
        "torso_pitch", "torso_pitch_t", "com_ref", "apa",
        "head_pos", "head_quat")}


def _clip_vec(vec, limit):
    norm = float(np.linalg.norm(vec))
    return vec * (limit / norm) if norm > limit else vec


class _TargetTracker:
    """二阶临界阻尼目标跟踪，输出位置/速度/加速度供 QP 前馈。

    相比一阶低通，二阶参考在目标跳变时给出连续的加速度，配合 `hand_vel/hand_acc`
    前馈可以显著减少跟踪滞后，同时避免速度阶跃造成的抖动。
    """

    def __init__(self, hands, bounds_rel=None):
        self.p0 = {k: np.array(v[0], dtype=float) for k, v in hands.items()}
        self.p = {k: self.p0[k].copy() for k in hands}
        self.v = {k: np.zeros(3) for k in hands}
        self.a = {k: np.zeros(3) for k in hands}
        self.q = {k: np.array(v[1], dtype=float) for k, v in hands.items()}
        self.w = {k: np.zeros(3) for k in hands}
        self.alpha = {k: np.zeros(3) for k in hands}
        self.bounds_rel = bounds_rel

    def _apply_bounds(self, key, bounds):
        if bounds is None:
            return
        for i, (lo, hi) in enumerate(bounds):
            if self.p[key][i] < lo:
                self.p[key][i] = lo
                if self.v[key][i] < 0.0:
                    self.v[key][i] = 0.0
            elif self.p[key][i] > hi:
                self.p[key][i] = hi
                if self.v[key][i] > 0.0:
                    self.v[key][i] = 0.0

    def _clamp_target(self, key, point, bounds):
        p = np.array(point, dtype=float)
        if bounds is not None:
            for i, (lo, hi) in enumerate(bounds):
                p[i] = min(max(p[i], lo), hi)
        if self.bounds_rel is not None:
            for i, (lo, hi) in enumerate(self.bounds_rel):
                p[i] = min(max(p[i], self.p0[key][i] + lo),
                           self.p0[key][i] + hi)
        return p

    def clamp_target(self, key, point, bounds):
        """对外暴露目标限幅，供预测守卫使用与实际目标一致的边界。"""
        return self._clamp_target(key, point, bounds)

    def update(self, raw, dt, omega, max_speed=None, governor=1.0,
               rot_tau=0.08, bounds=None, max_acc=30.0):
        out = {}
        max_v = None if max_speed is None else max_speed * governor
        for key in ("left_hand", "right_hand"):
            p_raw, q_raw = raw[key]
            # 先限制原始目标，再进滤波器；否则大拖动会在边界反复加速
            p_raw = self._clamp_target(key, p_raw, bounds)

            acc = (omega * omega * (p_raw - self.p[key])
                   - 2.0 * omega * self.v[key])
            acc = _clip_vec(acc, max_acc * max(governor, 0.15))
            v_old = self.v[key]
            v_new = v_old + acc * dt
            if max_v is not None:
                speed = float(np.linalg.norm(v_new))
                if speed > max_v:
                    v_new = v_new * (max_v / speed)
            self.a[key] = (v_new - v_old) / dt
            self.p[key] = self.p[key] + v_new * dt
            self.v[key] = v_new

            self._apply_bounds(key, bounds)
            if self.bounds_rel is not None:
                lo = self.p0[key] + np.array([b[0] for b in self.bounds_rel])
                hi = self.p0[key] + np.array([b[1] for b in self.bounds_rel])
                self._apply_bounds(key, tuple(zip(lo, hi)))

            alpha = 1.0 - np.exp(-dt / rot_tau)
            q_new = quat_slerp(self.q[key], np.asarray(q_raw, dtype=float), alpha)
            w_old = self.w[key].copy()
            self.w[key] = _clip_vec(quat_error(q_new, self.q[key]) / dt, 12.0)
            self.alpha[key] = (self.w[key] - w_old) / dt
            self.q[key] = q_new

            out[key] = dict(pos=self.p[key].copy(), vel=self.v[key].copy(),
                            acc=self.a[key].copy(), quat=self.q[key].copy(),
                            omega=self.w[key].copy(),
                            alpha=self.alpha[key].copy())
        return out


def _clamp_polygon(point, normals, offsets, margin):
    """把点投影到（内缩 margin 的）支撑多边形内。"""
    p = np.asarray(point, dtype=float).copy()
    for _ in range(8):
        violation = normals @ p - (offsets - margin)
        if violation.max() <= 0.0:
            break
        i = int(np.argmax(violation))
        p = p - violation[i] * normals[i]
    return p


def summarize(res, foot_anchor=None):
    """从 `run()` 的日志中提取便于横向对比的指标。"""
    log = res["log"]
    if not len(log["t"]):
        return {"steps": 0}
    base_z = log["base_pos"][:, 2]
    margin = log["margin"]
    p, pt = log["hand_pos"], log["hand_pos_t"]
    hand_err = np.concatenate([
        np.linalg.norm(p[:, :3] - pt[:, :3], axis=1),
        np.linalg.norm(p[:, 3:6] - pt[:, 3:6], axis=1),
    ])
    q, qt = log["hand_quat"], log["hand_quat_t"]
    rot_err = np.concatenate([
        2.0 * np.arccos(np.clip(np.abs(np.sum(q[:, 4 * i:4 * i + 4]
                                           * qt[:, 4 * i:4 * i + 4], axis=1)),
                               -1.0, 1.0))
        for i in range(2)
    ])
    out = {
        "steps": int(res["steps"]),
        "duration_s": float(log["t"][-1] + (log["t"][1] - log["t"][0])
                            if len(log["t"]) > 1 else 0.0),
        "base_z_min": float(base_z.min()),
        "margin_min_mm": float(margin.min() * 1e3),
        "hand_pos_rmse_mm": float(np.sqrt(np.mean(hand_err ** 2)) * 1e3),
        "hand_pos_max_mm": float(hand_err.max() * 1e3),
        "hand_rot_rmse_deg": float(np.degrees(np.sqrt(np.mean(rot_err ** 2)))),
        "qp_failures": int(sum(1 for s in res["status"] if s == "failed")),
        "qp_holds": int(sum(1 for s in res["status"] if s == "hold")),
        "fell": bool(base_z.min() < 0.45),
    }
    q = log["q_ref"]
    if len(log["t"]) > 4 and q.shape[0] > 4:
        dt_log = float(log["t"][1] - log["t"][0])
        jerk = np.diff(q, n=3, axis=0) / max(dt_log, 1e-6) ** 3
        out["q_jerk_rms"] = float(np.sqrt(np.mean(jerk ** 2)))
    if foot_anchor is not None:
        # foot_pos 是左右脚 6 维拼接；与初始锚点比较
        fp = log["foot_pos"]
        if fp.shape[1] >= 6 and len(foot_anchor) == 2:
            d = np.concatenate([
                np.linalg.norm(fp[:, :3] - foot_anchor[0][0], axis=1),
                np.linalg.norm(fp[:, 3:6] - foot_anchor[1][0], axis=1),
            ])
            out["foot_drift_max_mm"] = float(d.max() * 1e3)
    return out


# ---------------------------------------------------------------------- #
# 主循环
# ---------------------------------------------------------------------- #
def _run_kinematic(pipe, target_fn, base_z_fn=None, n_steps=None, viewer=None,
                   speed=1.0, verbose=True, pace=True, drive_markers=True,
                   target_omega=100.0, target_speed=None, margin_guard=0.0,
                   lean_gain=0.0, target_bounds=None, bounds_rel=None,
                   com_margin=0.015, rot_tau=0.08,
                   max_target_acc=30.0, predictive_guard=False, pred_buffer=0.008,
                   com_vel_gain=0.0, com_vel_tau=0.06,
                   max_track_err=0.5, log_every=1, balance=False,
                   governor=True):
    """实时闭环主循环。

    target_fn(step, t) -> {"left_hand": (pos, quat), "right_hand": (...)}
    base_z_fn(step, t) -> 骨盆高度参考

    target_omega: 目标二阶滤波带宽（rad/s）；轨迹模式用大值（近似透明），
                  mocap 拖动用 4–8 获得平滑加速。
    lean_gain:    CoM 配重增益；负值表示手向前时重心后坐（固定双脚下的稳定方向）。
    target_bounds: 拖动目标的绝对 [x/y/z] 工作空间边界，None 表示不限制。
    bounds_rel:   相对初始手位的平衡工作空间 (dx/dy/dz lo, hi)，超出后目标停在边界。
    predictive_guard: 用目标加速度预估所需 ZMP 偏移，提前降低目标速度/加速度。
    com_vel_gain: 真实 CoM 速度的阻尼增益（速度前馈系数）；mocap 拖动时抑制
                  前后方向接近 √(g/h) 的低频自激，轨迹模式保持 0。
    com_vel_tau:  上述速度估计的一阶低通时间常数，滤掉接触高频抖动。
    """
    r = pipe.robot
    m, d = r.model, r.data
    dt = pipe.dt
    n = pipe.n_steps if n_steps is None else n_steps
    if base_z_fn is None:
        base_z_fn = trajectory_base_z(pipe)

    # 用于预测手臂反力对 ZMP 的影响
    arm_mass = 0.0
    for name in ("left_shoulder_pitch_link", "right_shoulder_pitch_link"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            arm_mass += float(m.body_subtreemass[bid])
    total_mass = float(m.body_subtreemass[0])

    # 与 replay() 一致：从零速度开始，避免初始残速在高增益下放大
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0

    # 虚拟参考状态：与真实状态同源，之后只做全量积分
    dv = mujoco.MjData(m)
    dv.qpos[:] = d.qpos
    dv.qvel[:] = 0.0
    mujoco.mj_forward(m, dv)
    qp = WholeBodyQP(r, dv)

    posture = r.q_target.copy()
    kp = m.actuator_gainprm[r.body_act_ids, 0].copy()
    q_ref = dv.qpos[r.body_qpos_adr].copy()
    hand_init = {
        key: (d.site_xpos[r.site_ids[key]].copy(), r.site_quat(key))
        for key in ("left_hand", "right_hand")
    }
    tracker = _TargetTracker(hand_init, bounds_rel=bounds_rel)
    hand_nom = 0.5 * (hand_init["left_hand"][0] + hand_init["right_hand"][0])
    com_base = pipe.com_target.copy()
    normals, offsets = polygon_edges(pipe.support_poly)
    foot_radius = float(m.geom_size[r.foot_contact_geoms[0], 0])
    offsets_eff = offsets - foot_radius - config.SUPPORT_SHRINK
    sync_every = max(1, int(round(_VIEWER_SYNC_DT / dt)))
    governor_smooth = 1.0
    jac_com_rt = np.zeros((3, m.nv))
    com_vel_f = np.zeros(2)
    balance_a = np.zeros(2)      # 双手平均加速度（低通）
    balance_ad = np.zeros(2)     # 左右手加速度差（低通）
    use_traj_ref = target_omega >= 50.0
    if use_traj_ref:
        # 在线调速会把轨迹时间拉长；固定 n_steps 会在动作未走完时截断。
        # 允许最多 3 倍时长，并在轨迹时钟走完 pre+duration+post 后提前退出。
        n = max(n, int(round(3.0 * pipe.n_steps)))
    traj_time = 0.0
    traj_speed = 1.0
    governor_on = bool(governor)
    log = _empty_log()
    status = []
    steps_done = 0
    t_start = time.perf_counter()

    for step in range(n):
        if viewer is not None and not viewer.is_running():
            break

        t = step * dt
        # 不额外调用 mj_forward：它会用旧 ctrl 预热接触求解器，与 replay() 不一致
        if use_traj_ref:
            if governor_on:
                # 轨迹模式：CoM/捕获点接近支撑边界时对新轨迹时间做调速
                mujoco.mj_jacSubtreeCom(m, d, jac_com_rt, 0)
                vcom = jac_com_rt[:, r.ik_dof_cols] @ d.qvel[r.ik_dof_cols]
                com_now = d.subtree_com[0][:2]
                xi = com_now + vcom[:2] / config.WBC_OMEGA_LIP
                margin = min(r.support_margin(com_now), r.support_margin(xi))
                s_target = float(np.clip(margin / config.KIN_GOV_MARGIN, 0.05, 1.0))
                tau_s = (config.KIN_GOV_TAU_UP if s_target > traj_speed
                         else config.KIN_GOV_TAU_DOWN)
                rate_s = 1.0 - np.exp(-dt / tau_s)
                traj_speed += rate_s * (s_target - traj_speed)
                traj_speed = float(np.clip(traj_speed, 0.05, 1.0))
                traj_time += dt * traj_speed
            else:
                # 评估模式：严格按源数据时钟执行，不做任何时间缩放/封顶
                traj_speed = 1.0
                traj_time += dt
            tt = float(np.clip(traj_time - pipe.pre, 0.0, pipe.traj.duration))
            raw = pipe.traj.hand(tt)
            base_z_ref = pipe.traj.base_z(tt)
            governor = traj_speed
        else:
            raw = target_fn(step, t)
            base_z_ref = base_z_fn(step, t)

        # 平衡守卫：裕度越小，允许的目标速度越低（平滑减速而不是硬冻结）
        governor = 1.0
        if margin_guard > 0.0:
            governor = float(np.clip(
                r.support_margin(d.subtree_com[0][:2]) / margin_guard, 0.0, 1.0))

        # 预测守卫：用当前目标加速度估计所需 ZMP 偏移，提前减速而不是等裕度耗尽
        if predictive_guard and arm_mass > 0.0:
            a_est = np.zeros(3)
            for key in ("left_hand", "right_hand"):
                p_raw = tracker.clamp_target(key, raw[key][0], target_bounds)
                a_est += (target_omega ** 2 * (p_raw - tracker.p[key])
                          - 2.0 * target_omega * tracker.v[key])
            a_est *= 0.5 * arm_mass / total_mass
            h_com = float(d.subtree_com[0][2])
            delta_zmp = (h_com / abs(float(m.opt.gravity[2]))) * a_est[:2]
            avail = offsets_eff - normals @ d.subtree_com[0][:2]
            need = normals @ delta_zmp
            scale = 1.0
            for avail_i, need_i in zip(avail, need):
                if need_i > 1e-6:
                    scale = min(scale, max(0.0, (avail_i - pred_buffer) / need_i))
            governor = min(governor, float(np.clip(scale, 0.0, 1.0)))

        # 平滑守卫系数：危险时快速收紧，恢复时缓慢放开，避免目标速度阶跃造成抖动
        rate = 0.30 if governor < governor_smooth else 0.02
        governor_smooth += rate * (governor - governor_smooth)
        governor = governor_smooth
        if use_traj_ref:
            governor = traj_speed

        tracked = tracker.update(raw, dt, target_omega,
                                 max_speed=target_speed, governor=governor,
                                 rot_tau=rot_tau, bounds=target_bounds,
                                 max_acc=max_target_acc)
        target = {key: (tracked[key]["pos"], tracked[key]["quat"])
                  for key in ("left_hand", "right_hand")}

        # CoM 参考
        if balance:
            # 预期性姿势调整（APA）：手臂加速时提前把 CoM 移向反作用方向，
            # 给固定的双脚争取 CoP 裕度；真实 CoM 速度用于抑制倒立摆慢发散。
            if use_traj_ref:
                preview = pipe.traj.hand_ref(
                    min(tt + config.BAL_APA_LEAD, pipe.traj.duration))
                a_left = preview["left_hand"]["acc"][:2]
                a_right = preview["right_hand"]["acc"][:2]
            else:
                a_left = tracked["left_hand"]["acc"][:2]
                a_right = tracked["right_hand"]["acc"][:2]
            alpha = 1.0 - np.exp(-dt / config.BAL_ACC_TAU)
            balance_a += alpha * (0.5 * (a_left + a_right) - balance_a)
            balance_ad += alpha * (0.5 * (a_right - a_left) - balance_ad)
            mujoco.mj_jacSubtreeCom(m, d, jac_com_rt, 0)
            alpha_v = 1.0 - np.exp(-dt / com_vel_tau)
            com_vel_f += alpha_v * ((jac_com_rt @ d.qvel)[:2] - com_vel_f)
            h_com = float(d.subtree_com[0][2])
            arm_gain = (arm_mass / max(total_mass, 1e-9)
                        * h_com / abs(float(m.opt.gravity[2])))
            com_ref = _clamp_polygon(
                com_base + config.BAL_APA_GAIN * arm_gain * balance_a
                - config.BAL_VEL_DAMP * com_vel_f,
                normals, offsets_eff, config.BAL_COM_MARGIN)
        elif abs(lean_gain) > 0.0:
            # CoM 参考：随手部目标前倾/侧移，并限制在支撑多边形内
            mean_target = 0.5 * (tracked["left_hand"]["pos"]
                                 + tracked["right_hand"]["pos"])
            com_ref = _clamp_polygon(
                com_base + lean_gain * (mean_target - hand_nom)[:2],
                normals, offsets_eff, com_margin)
        else:
            com_ref = com_base

        torso_target = pipe.traj.torso_quat(tt) if use_traj_ref else None
        if balance and torso_target is not None:
            # 左右手加速度差对应躯干的滚转/俯仰角动量，躯干反向转动来吸收
            torso_target = quat_mul(
                quat_from_rpy(-config.BAL_TORSO_DIFF_GAIN * balance_ad[1],
                              -config.BAL_TORSO_DIFF_GAIN * balance_ad[0], 0.0),
                torso_target)

        # 跟踪误差守卫：伺服明显跟不上时冻结参考，避免参考跑飞
        qa = d.qpos[r.body_qpos_adr]
        track_err = float(np.abs(q_ref - qa).max())
        hold = track_err > max_track_err

        if not hold:
            dv.qvel[:] = 0.0
            mujoco.mj_forward(m, dv)
            com_vel_ff_rt = None
            if com_vel_gain > 0.0 and not balance:
                # QP 在零速度的虚拟参考上求解，阻尼必须用真实 CoM 速度；
                # 先低通滤掉接触高频抖动，避免把噪声放大成参考抖动。
                mujoco.mj_jacSubtreeCom(m, d, jac_com_rt, 0)
                a_f = 1.0 - np.exp(-dt / com_vel_tau)
                com_vel_f += a_f * ((jac_com_rt @ d.qvel)[:2] - com_vel_f)
                com_vel_ff_rt = -com_vel_gain * com_vel_f
            dq = qp.solve(
                target, base_z_ref, com_ref, pipe.support_poly,
                posture,
                support_lookahead=pipe.spec.support_lookahead,
                arm_vel_limit=pipe.spec.arm_vel_limit,
                base_vel_limit=pipe.spec.base_vel_limit,
                base_angvel_limit=pipe.spec.base_angvel_limit,
                hand_vel={k: tracked[k]["vel"] for k in target},
                hand_acc={k: tracked[k]["acc"] for k in target},
                hand_omega={k: tracked[k]["omega"] for k in target},
                com_vel_ff=com_vel_ff_rt,
                torso_quat_target=torso_target,
            )
            if qp.last_status == "solved":
                full = np.zeros(m.nv)
                full[qp.cols] = dq
                mujoco.mj_integratePos(m, dv.qpos, full, 1.0)
                q_ref = dv.qpos[r.body_qpos_adr].copy()
                status.append("solved")
            else:
                status.append("failed")
                dq = None
        else:
            status.append("hold")
            mujoco.mj_forward(m, dv)

        # 参考状态上的重力/偏置力矩前馈
        ff = np.clip(dv.qfrc_bias[r.body_dof_adr] / kp,
                     -config.REPLAY_FF_LIMIT, config.REPLAY_FF_LIMIT)
        ramp = min(1.0, step * dt / 0.3)
        d.ctrl[r.body_act_ids] = (q_ref + ramp * ff
                                  + pipe.spec.replay_k * (q_ref - qa))
        d.ctrl[r.finger_act_ids] = r.finger_ctrl

        if drive_markers:
            for key in ("left_hand", "right_hand"):
                mid = pipe.mocap_ids[key]
                d.mocap_pos[mid] = target[key][0]
                d.mocap_quat[mid] = target[key][1]

        if step % log_every == 0:
            log["t"].append(t)
            log["hand_pos_t"].append(np.concatenate([target["left_hand"][0],
                                                     target["right_hand"][0]]))
            log["hand_quat_t"].append(np.concatenate([target["left_hand"][1],
                                                      target["right_hand"][1]]))
            log["hand_pos"].append(np.concatenate([d.site_xpos[r.site_ids["left_hand"]],
                                                   d.site_xpos[r.site_ids["right_hand"]]]))
            log["hand_quat"].append(np.concatenate([r.site_quat("left_hand"),
                                                    r.site_quat("right_hand")]))
            log["foot_pos"].append(np.concatenate([d.site_xpos[r.site_ids["left_foot"]],
                                                   d.site_xpos[r.site_ids["right_foot"]]]))
            log["foot_quat"].append(np.concatenate([r.site_quat("left_foot"),
                                                    r.site_quat("right_foot")]))
            log["base_pos"].append(d.qpos[:3].copy())
            log["qpos"].append(d.qpos.copy())
            log["qvel"].append(d.qvel.copy())
            log["ctrl"].append(d.ctrl.copy())
            log["q_ref"].append(q_ref.copy())
            log["com"].append(d.subtree_com[0].copy())
            log["ncon"].append(d.ncon)
            log["margin"].append(r.support_margin(d.subtree_com[0][:2]))
            log["com_ref"].append(com_ref.copy())
            log["apa"].append(balance_a.copy())
            log["head_pos"].append(r.head_pos())
            log["head_quat"].append(r.head_quat())
            log["governor"].append(governor)
            log["solve_time"].append(qp.solve_time if not hold else 0.0)
            log["status"].append(status[-1])
            log["torso_pitch"].append(torso_pitch_deg(r, d))
            log["torso_pitch_t"].append(
                float(np.degrees(pipe.traj.torso_pitch(tt)[0]))
                if use_traj_ref else 0.0)

        mujoco.mj_step(m, d)
        steps_done += 1

        if (use_traj_ref
                and traj_time >= pipe.pre + pipe.traj.duration + pipe.post):
            break

        if viewer is not None and step % sync_every == 0:
            viewer.sync()
        if pace and viewer is not None:
            delay = t_start + (step + 1) * dt / max(speed, 0.05) - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)

        if verbose and step % 1000 == 0:
            pe = max(np.linalg.norm(log["hand_pos"][-1][3 * i:3 * i + 3]
                                    - log["hand_pos_t"][-1][3 * i:3 * i + 3])
                     for i in range(2))
            print(f"  [rt] t={t:5.2f}s  hand_err={pe:.3f}m  "
                  f"qp={qp.solve_time*1e3:.2f}ms  margin={log['margin'][-1]:+.3f}  "
                  f"ncon={d.ncon}  status={status[-1]}", flush=True)

    if viewer is not None:
        viewer.sync()
    wall = time.perf_counter() - t_start
    for k, v in log.items():
        log[k] = np.asarray(v)
    return {
        "log": log,
        "solve_time": np.asarray(log["solve_time"]),
        "status": status,
        "wall_time": wall,
        "steps": steps_done,
    }


# ---------------------------------------------------------------------- #
# 动力学级 WBC 主循环
# ---------------------------------------------------------------------- #
def plan_head_positions(pipe, plan):
    """用规划 qpos 正运动学预计算头部点位置（与实时同一时间轴）。"""
    r = pipe.robot
    m = r.model
    dp = mujoco.MjData(m)
    out = np.zeros((len(plan.times), 3))
    for i in range(len(plan.times)):
        dp.qpos[:7] = plan.qpos[i, :7]
        dp.qpos[r.body_qpos_adr] = plan.qpos[i, 7:]
        mujoco.mj_forward(m, dp)
        out[i] = dp.geom_xpos[r.head_geom]
    return out


def run(pipe, target_fn, base_z_fn=None, n_steps=None, viewer=None,
        speed=1.0, verbose=True, pace=True, drive_markers=True,
        target_omega=100.0, target_speed=None, margin_guard=0.0,
        lean_gain=0.0, target_bounds=None, bounds_rel=None,
        com_margin=0.015, rot_tau=0.08,
        max_target_acc=30.0, predictive_guard=False, pred_buffer=0.008,
        com_vel_gain=0.0, com_vel_tau=0.06,
        max_track_err=0.5, log_every=1, controller="kinematic",
        governor=None, control_every=1, sync_dt=_VIEWER_SYNC_DT):
    """统一入口。

    controller:
      - `"kinematic"`：位置伺服回放（稳定基线，默认开调速守卫）；
      - `"balance"`：平衡前馈控制器（APA + CoM 速度阻尼 + 躯干角动量补偿），
        默认严格按源数据时钟执行，不做时间缩放；
      - `"wbc"`：实验性动力学 WBC。
    """
    if controller in ("kinematic", "balance"):
        balance = controller == "balance"
        if governor is None:
            governor = not balance
        return _run_kinematic(
            pipe, target_fn, base_z_fn=base_z_fn, n_steps=n_steps,
            viewer=viewer, speed=speed, verbose=verbose, pace=pace,
            drive_markers=drive_markers, target_omega=target_omega,
            target_speed=target_speed, margin_guard=margin_guard,
            lean_gain=lean_gain, target_bounds=target_bounds,
            bounds_rel=bounds_rel, com_margin=com_margin, rot_tau=rot_tau,
            max_target_acc=max_target_acc, predictive_guard=predictive_guard,
            pred_buffer=pred_buffer, com_vel_gain=com_vel_gain,
            com_vel_tau=com_vel_tau, max_track_err=max_track_err,
            log_every=log_every, balance=balance, governor=governor)
    return _run_wbc(
        pipe, target_fn, base_z_fn=base_z_fn, n_steps=n_steps,
        viewer=viewer, speed=speed, verbose=verbose, pace=pace,
        drive_markers=drive_markers, target_omega=target_omega,
        target_speed=target_speed, margin_guard=margin_guard,
        lean_gain=lean_gain, target_bounds=target_bounds,
        bounds_rel=bounds_rel, com_margin=com_margin, rot_tau=rot_tau,
        max_target_acc=max_target_acc, predictive_guard=predictive_guard,
        pred_buffer=pred_buffer, max_track_err=max_track_err,
        log_every=log_every, control_every=control_every, sync_dt=sync_dt)


def _run_wbc(pipe, target_fn, base_z_fn=None, n_steps=None, viewer=None,
             speed=1.0, verbose=True, pace=True, drive_markers=True,
             target_omega=100.0, target_speed=None, margin_guard=0.0,
             lean_gain=0.0, target_bounds=None, bounds_rel=None,
             com_margin=0.015, rot_tau=0.08,
             max_target_acc=30.0, predictive_guard=False, pred_buffer=0.008,
             max_track_err=0.5, log_every=1, control_every=1,
             sync_dt=_VIEWER_SYNC_DT):
    r = pipe.robot
    m, d = r.model, r.data
    dt = pipe.dt
    n = pipe.n_steps if n_steps is None else n_steps
    if base_z_fn is None:
        base_z_fn = trajectory_base_z(pipe)

    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(m, d)

    wbc = WholeBodyWBC(r, d)
    posture = r.q_target.copy()
    com_base = pipe.com_target.copy()
    normals, offsets = polygon_edges(pipe.support_poly)
    foot_radius = float(m.geom_size[r.foot_contact_geoms[0], 0])
    offsets_eff = offsets - foot_radius - config.SUPPORT_SHRINK

    hand_init = {
        key: (d.site_xpos[r.site_ids[key]].copy(), r.site_quat(key))
        for key in ("left_hand", "right_hand")
    }
    tracker = _TargetTracker(hand_init, bounds_rel=bounds_rel)
    use_traj_ref = target_omega >= 50.0

    sync_every = max(1, int(round(sync_dt / dt)))

    # 头部参考：有离线规划时用规划头部轨迹（包含任务本身要求的弯腰/下蹲），
    # 没有规划（mocap/在线模式）时保持初始头部位置。
    head_ref_traj = None
    plan_result = getattr(pipe, "plan_result", None)
    if plan_result is not None:
        head_ref_traj = getattr(pipe, "plan_head", None)
        if head_ref_traj is None:
            head_ref_traj = plan_head_positions(pipe, plan_result)
    head_ref0 = r.head_pos()[:2].copy()
    base_quat0 = d.qpos[3:7].copy()
    base_xy0 = d.qpos[:2].copy()
    base_xy_traj = None
    if plan_result is not None:
        base_xy_traj = plan_result.qpos[:, :2]

    # 与 balance 相同的平衡前馈状态
    arm_mass = 0.0
    for name in ("left_shoulder_pitch_link", "right_shoulder_pitch_link"):
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            arm_mass += float(m.body_subtreemass[bid])
    total_mass = float(m.body_subtreemass[0])
    balance_a = np.zeros(2)
    balance_ad = np.zeros(2)
    com_vel_f = np.zeros(2)
    jac_com_rt = np.zeros((3, m.nv))
    log = _empty_log()
    status = []
    steps_done = 0
    tau_held = None
    t_start = time.perf_counter()

    for step in range(n):
        if viewer is not None and not viewer.is_running():
            break

        t = step * dt
        tt = float(np.clip(t - pipe.pre, 0.0, pipe.traj.duration))
        torso_ref = None
        head_gate = 1.0

        if use_traj_ref:
            # 严格按源数据时钟：不做时间缩放，也不做目标幅度回抽。
            lean = abs(float(pipe.traj.torso_pitch(tt)[0]))
            if head_ref_traj is not None:
                idx = min(step, len(head_ref_traj) - 1)
                planned = head_ref_traj[idx, :2]
                # 直立任务锁住头部水平位置；需要弯腰时平滑过渡到规划头部轨迹
                follow = float(np.clip(
                    np.degrees(lean) / config.WBC_HEAD_FOLLOW_DEG,
                    config.WBC_HEAD_PLAN_FOLLOW, 1.0))
                head_ref = head_ref0 + follow * (planned - head_ref0)
            else:
                head_gate = float(np.clip(
                    1.0 - lean / np.radians(config.WBC_HEAD_LEAN_GATE_DEG),
                    0.0, 1.0))
                head_ref = head_ref0
            hand_ref = pipe.traj.hand_ref(tt)
            hand_pos_t = np.concatenate([hand_ref[k]["pos"]
                                         for k in ("left_hand", "right_hand")])
            hand_quat_t = np.concatenate([hand_ref[k]["quat"]
                                          for k in ("left_hand", "right_hand")])
            base_z_ref = pipe.traj.base_z(tt)
            torso_ref = pipe.traj.torso_quat(tt)
            preview = pipe.traj.hand_ref(
                min(tt + config.BAL_APA_LEAD, pipe.traj.duration))
            a_left = preview["left_hand"]["acc"][:2]
            a_right = preview["right_hand"]["acc"][:2]
        else:
            base_z_ref = base_z_fn(step, t)
            raw = target_fn(step, t)
            tracked = tracker.update(raw, dt, target_omega,
                                     max_speed=target_speed, governor=1.0,
                                     rot_tau=rot_tau, bounds=target_bounds,
                                     max_acc=max_target_acc)
            hand_ref = {
                key: dict(pos=tracked[key]["pos"], vel=tracked[key]["vel"],
                          acc=tracked[key]["acc"], quat=tracked[key]["quat"],
                          omega=tracked[key]["omega"],
                          alpha=tracked[key].get("alpha", np.zeros(3)))
                for key in ("left_hand", "right_hand")
            }
            hand_pos_t = np.concatenate([hand_ref[k]["pos"]
                                         for k in ("left_hand", "right_hand")])
            hand_quat_t = np.concatenate([hand_ref[k]["quat"]
                                          for k in ("left_hand", "right_hand")])
            head_ref = head_ref0
            a_left = tracked["left_hand"]["acc"][:2]
            a_right = tracked["right_hand"]["acc"][:2]

        # APA（预期性姿势调整）：用 0.15 s 后的手部加速度把 CoM 参考提前
        # 移向反作用方向；左右手加速度差用于躯干角动量补偿。
        alpha_a = 1.0 - np.exp(-dt / config.BAL_ACC_TAU)
        balance_a += alpha_a * (0.5 * (a_left + a_right) - balance_a)
        balance_ad += alpha_a * (0.5 * (a_right - a_left) - balance_ad)
        mujoco.mj_jacSubtreeCom(m, d, jac_com_rt, 0)
        alpha_v = 1.0 - np.exp(-dt / config.BAL_VEL_DAMP_TAU)
        com_vel_f += alpha_v * ((jac_com_rt @ d.qvel)[:2] - com_vel_f)
        h_com = float(d.subtree_com[0][2])
        arm_gain = (arm_mass / max(total_mass, 1e-9)
                    * h_com / abs(float(m.opt.gravity[2])))
        com_ref = _clamp_polygon(
            com_base + config.BAL_APA_GAIN * arm_gain * balance_a,
            normals, offsets_eff, config.BAL_COM_MARGIN)
        if torso_ref is not None:
            torso_ref = quat_mul(
                quat_from_rpy(-config.BAL_TORSO_DIFF_GAIN * balance_ad[1],
                              -config.BAL_TORSO_DIFF_GAIN * balance_ad[0], 0.0),
                torso_ref)

        # 控制降频：WBC 每个 control_every 步解一次，其余步保持上一拍力矩。
        # 物理仍按 500 Hz 步进，默认 control_every=1 时行为完全不变。
        if step % control_every == 0 or tau_held is None:
            tau = wbc.solve(hand_ref, com_ref, base_z_ref, posture,
                            torso_ref=torso_ref, head_ref=head_ref,
                            head_gate=head_gate, base_quat_ref=base_quat0,
                            base_xy_ref=(base_xy_traj[min(step, len(base_xy_traj) - 1)]
                                         if base_xy_traj is not None else base_xy0))
            if tau is not None:
                tau_held = tau
                wbc.apply(tau)
                status.append("solved")
            else:
                tau_held = None
                # 回退：至少补偿重力/科氏力，避免力矩归零后软腿
                wbc.apply(d.qfrc_bias[r.body_dof_adr].copy())
                status.append("failed")
        else:
            wbc.apply(tau_held)
            status.append("solved")

        if drive_markers:
            for key in ("left_hand", "right_hand"):
                mid = pipe.mocap_ids[key]
                d.mocap_pos[mid] = hand_ref[key]["pos"]
                d.mocap_quat[mid] = hand_ref[key]["quat"]

        if step % log_every == 0:
            log["t"].append(t)
            log["hand_pos_t"].append(hand_pos_t)
            log["hand_quat_t"].append(hand_quat_t)
            log["hand_pos"].append(np.concatenate([
                d.site_xpos[r.site_ids["left_hand"]],
                d.site_xpos[r.site_ids["right_hand"]]]))
            log["hand_quat"].append(np.concatenate([
                r.site_quat("left_hand"), r.site_quat("right_hand")]))
            log["foot_pos"].append(np.concatenate([
                d.site_xpos[r.site_ids["left_foot"]],
                d.site_xpos[r.site_ids["right_foot"]]]))
            log["foot_quat"].append(np.concatenate([
                r.site_quat("left_foot"), r.site_quat("right_foot")]))
            log["base_pos"].append(d.qpos[:3].copy())
            log["qpos"].append(d.qpos.copy())
            log["qvel"].append(d.qvel.copy())
            log["ctrl"].append(d.ctrl.copy())
            log["q_ref"].append(d.qpos[r.body_qpos_adr].copy())
            log["com"].append(d.subtree_com[0].copy())
            log["ncon"].append(d.ncon)
            log["margin"].append(r.support_margin(d.subtree_com[0][:2]))
            log["com_ref"].append(np.asarray(com_ref, dtype=float).copy())
            log["apa"].append(balance_a.copy())
            log["head_pos"].append(r.head_pos())
            log["head_quat"].append(r.head_quat())
            log["governor"].append(1.0)
            log["torso_pitch"].append(torso_pitch_deg(r, d))
            log["torso_pitch_t"].append(
                float(np.degrees(pipe.traj.torso_pitch(tt)[0]))
                if use_traj_ref else 0.0)
            log["solve_time"].append(wbc.solve_time)
            log["status"].append(status[-1])

        mujoco.mj_step(m, d)
        steps_done += 1

        if viewer is not None and step % sync_every == 0:
            viewer.sync()
        if pace and viewer is not None:
            delay = t_start + (step + 1) * dt / max(speed, 0.05) - time.perf_counter()
            if delay > 0.0:
                time.sleep(delay)

        if verbose and step % 1000 == 0:
            pe = max(np.linalg.norm(log["hand_pos"][-1][3 * i:3 * i + 3]
                                    - log["hand_pos_t"][-1][3 * i:3 * i + 3])
                     for i in range(2))
            print(f"  [wbc] t={t:5.2f}s  hand_err={pe:.3f}m  "
                  f"qp={wbc.solve_time*1e3:.2f}ms  margin={log['margin'][-1]:+.3f}  "
                  f"ncon={d.ncon}  status={status[-1]}", flush=True)

    if viewer is not None:
        viewer.sync()
    wall = time.perf_counter() - t_start
    for k, v in log.items():
        log[k] = np.asarray(v)
    return {
        "log": log,
        "solve_time": np.asarray(log["solve_time"]),
        "status": status,
        "wall_time": wall,
        "steps": steps_done,
    }
