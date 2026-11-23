"""双手 6-DOF 目标轨迹：C2 连续插值 + 相对初始手姿态的 RPY 偏移。"""

import numpy as np
from scipy.interpolate import CubicSpline

from .scenarios import Scenario, get_scenario
from .utils import quat_error, quat_mul


def quat_from_rpy(roll, pitch, yaw):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll) 对应的四元数 (w, x, y, z)。"""
    cr, sr = np.cos(roll / 2.0), np.sin(roll / 2.0)
    cp, sp = np.cos(pitch / 2.0), np.sin(pitch / 2.0)
    cy, sy = np.cos(yaw / 2.0), np.sin(yaw / 2.0)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ])


class ClampedCubicSpline:
    """端点速度为零的三次样条，C2 连续，用于生成平滑轨迹。"""

    def __init__(self, times, values):
        self.times = np.asarray(times, float)
        self.values = np.asarray(values, float)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        self.spline = CubicSpline(self.times, self.values, axis=0,
                                  bc_type=((1, np.zeros(self.values.shape[1])),
                                           (1, np.zeros(self.values.shape[1]))))
        self._d1 = self.spline.derivative(1)
        self._d2 = self.spline.derivative(2)
        self.duration = float(self.times[-1])

    def __call__(self, t):
        return self.spline(np.clip(t, self.times[0], self.times[-1]))

    def deriv(self, t, order=1):
        t = float(np.clip(t, self.times[0], self.times[-1]))
        if order == 1:
            return self._d1(t)
        if order == 2:
            return self._d2(t)
        return np.zeros_like(self.values[0])


def _resolve_pos(rows, anchor):
    """None 行表示初始手位置。"""
    return np.asarray([anchor if r is None else r for r in rows], float)


class MinJerkSpline:
    """逐段最小 jerk（五次多项式）插值。

    每个关键帧处速度、加速度均为零，因此不会像三次样条那样在“停顿点”
    过冲，适合快速到达/折返的手部轨迹。
    """

    def __init__(self, times, values):
        self.times = np.asarray(times, float)
        self.values = np.asarray(values, float)
        if self.values.ndim == 1:
            self.values = self.values[:, None]
        self.duration = float(self.times[-1])

    def _segment(self, t):
        t = float(np.clip(t, self.times[0], self.times[-1]))
        i = int(np.searchsorted(self.times, t, side="right") - 1)
        i = min(max(i, 0), len(self.times) - 2)
        span = float(self.times[i + 1] - self.times[i])
        return i, (t - self.times[i]) / span, span

    def __call__(self, t):
        i, u, _ = self._segment(t)
        h = 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
        return self.values[i] + (self.values[i + 1] - self.values[i]) * h

    def deriv(self, t, order=1):
        i, u, span = self._segment(t)
        diff = self.values[i + 1] - self.values[i]
        if order == 1:
            return diff * (30.0 * u**2 * (1.0 - u) ** 2) / span
        if order == 2:
            return diff * (60.0 * u * (1.0 - u) * (1.0 - 2.0 * u)) / span**2
        if order == 3:
            return diff * (60.0 - 360.0 * u + 360.0 * u**2) / span**3
        return np.zeros_like(self.values[0])


class BimanualTrajectory:
    """双手位置 + 姿态（相对初始姿态的 RPY 偏移）+ 骨盆高度参考。"""

    def __init__(self, robot, scenario="demo"):
        spec = get_scenario(scenario) if isinstance(scenario, str) else scenario
        self.spec = spec
        n = len(spec.times)
        for name in ("left_pos", "left_rpy_deg"):
            if len(getattr(spec, name)) != n:
                raise ValueError(f"场景 {spec.key}: {name} 长度与 times 不一致")
        for name in ("right_pos", "right_rpy_deg"):
            rows = getattr(spec, name)
            if rows is not None and len(rows) != n:
                raise ValueError(f"场景 {spec.key}: {name} 长度与 times 不一致")
        if spec.base_dz and len(spec.base_dz) != n:
            raise ValueError(f"场景 {spec.key}: base_dz 长度与 times 不一致")
        if spec.torso_pitch_deg and len(spec.torso_pitch_deg) != n:
            raise ValueError(f"场景 {spec.key}: torso_pitch_deg 长度与 times 不一致")
        p0l, q0l = robot.hand_anchor["left_hand"]
        p0r, q0r = robot.hand_anchor["right_hand"]
        z0 = float(robot.data.qpos[2])
        self.q0t = np.asarray(robot.torso_quat, float)

        pos_l = _resolve_pos(spec.left_pos, p0l)
        if spec.right_pos is None:
            pos_r = pos_l.copy()
            pos_r[:, 1] *= -1.0
        else:
            pos_r = _resolve_pos(spec.right_pos, p0r)

        rpy_l = np.deg2rad(np.asarray(spec.left_rpy_deg, float))
        if spec.right_rpy_deg is None:
            rpy_r = rpy_l.copy()
            rpy_r[:, 0] *= -1.0
            rpy_r[:, 2] *= -1.0
        else:
            rpy_r = np.deg2rad(np.asarray(spec.right_rpy_deg, float))

        dz = np.asarray(spec.base_dz, float)
        if dz.size == 0:
            dz = np.zeros(len(spec.times))
        base_z = z0 + dz

        pitch = np.asarray(spec.torso_pitch_deg, float)
        if pitch.size == 0:
            pitch = np.zeros(len(spec.times))
        pitch = np.deg2rad(pitch)

        times = np.asarray(spec.times, float) * spec.time_scale

        spline = MinJerkSpline if spec.interp == "minjerk" else ClampedCubicSpline
        self.pos_l = spline(times, pos_l)
        self.pos_r = spline(times, pos_r)
        self.rpy_l = spline(times, rpy_l)
        self.rpy_r = spline(times, rpy_r)
        self.z = spline(times, base_z[:, None])
        self.torso_pitch = spline(times, pitch[:, None])
        self.q0l, self.q0r = q0l, q0r
        self.duration = float(times[-1])
        self.key_times = times

        # 首/末段横向外让：sin² 形状，两端值为零且一阶导数为零，
        # 不改变折返点处的速度，只把“从自然下垂位出发/回到自然下垂位”的
        # 那一段路径向身体外侧推出去，避免手在髋部高度扫过躯干。
        self.reach_lateral = float(spec.reach_lateral)
        self._lateral_spans = []
        if self.reach_lateral > 0.0 and len(times) >= 3:
            self._lateral_spans = [(float(times[0]), float(times[1])),
                                   (float(times[-2]), float(times[-1]))]

    def _lateral(self, t):
        """返回 (外让量, 一阶导, 二阶导)，左臂 +y、右臂 -y。"""
        if not self._lateral_spans:
            return 0.0, 0.0, 0.0
        amp, d1, d2 = 0.0, 0.0, 0.0
        for a, b in self._lateral_spans:
            if a <= t <= b and b > a:
                u = (t - a) / (b - a)
                w = np.pi / (b - a)
                amp += self.reach_lateral * np.sin(np.pi * u) ** 2
                d1 += self.reach_lateral * w * np.sin(2.0 * np.pi * u)
                d2 += self.reach_lateral * 2.0 * w * w * np.cos(2.0 * np.pi * u)
        return amp, d1, d2

    def hand(self, t):
        t = float(np.clip(t, 0.0, self.duration))
        side = self._lateral(t)[0]
        pos_l = np.asarray(self.pos_l(t), float).copy()
        pos_r = np.asarray(self.pos_r(t), float).copy()
        pos_l[1] += side
        pos_r[1] -= side
        return {
            "left_hand": (pos_l,
                          quat_mul(quat_from_rpy(*self.rpy_l(t)), self.q0l)),
            "right_hand": (pos_r,
                           quat_mul(quat_from_rpy(*self.rpy_r(t)), self.q0r)),
        }

    def _quat_omega_alpha(self, spline, q0, t):
        """由 RPY 样条得到四元数、世界系角速度与角加速度。"""
        h = 1e-4

        def q_at(tt):
            tt = float(np.clip(tt, 0.0, self.duration))
            return quat_mul(quat_from_rpy(*spline(tt)), q0)

        def omega_at(tt):
            t0 = max(0.0, tt - h)
            t1 = min(self.duration, tt + h)
            if t1 - t0 < 1e-9:
                return np.zeros(3)
            return quat_error(q_at(t1), q_at(t0)) / (t1 - t0)

        omega = omega_at(t)
        ta = max(0.0, t - h)
        tb = min(self.duration, t + h)
        alpha = (omega_at(tb) - omega_at(ta)) / max(tb - ta, 1e-9)
        return q_at(t), omega, alpha

    def hand_ref(self, t):
        """带位置/速度/加速度、姿态/角速度/角加速度的 6-DOF 参考。"""
        t = float(np.clip(t, 0.0, self.duration))
        side, side_d1, side_d2 = self._lateral(t)
        out = {}
        for key, pos_s, rpy_s, q0, sign in (
            ("left_hand", self.pos_l, self.rpy_l, self.q0l, 1.0),
            ("right_hand", self.pos_r, self.rpy_r, self.q0r, -1.0),
        ):
            quat, omega, alpha = self._quat_omega_alpha(rpy_s, q0, t)
            pos = np.asarray(pos_s(t), float).copy()
            vel = np.asarray(pos_s.deriv(t, 1), float).copy()
            acc = np.asarray(pos_s.deriv(t, 2), float).copy()
            pos[1] += sign * side
            vel[1] += sign * side_d1
            acc[1] += sign * side_d2
            out[key] = dict(pos=pos, vel=vel, acc=acc,
                            quat=quat, omega=omega, alpha=alpha)
        return out

    def base_z(self, t):
        t = float(np.clip(t, 0.0, self.duration))
        return float(self.z(t)[0])

    def torso_quat(self, t):
        """躯干世界系目标姿态：初始直立姿态叠加绕世界 y 轴的前倾角。"""
        t = float(np.clip(t, 0.0, self.duration))
        pitch = float(self.torso_pitch(t)[0])
        return quat_mul(quat_from_rpy(0.0, pitch, 0.0), self.q0t)
