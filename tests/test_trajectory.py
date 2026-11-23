"""轨迹插值与目标手速的数值检查（用桩对象，不加载 MuJoCo 模型）。"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g1_wbc.scenarios import get_scenario
from g1_wbc.trajectory import BimanualTrajectory, MinJerkSpline

# demo 起始手位置（自然下垂），姿态为单位四元数
_P0L = np.array([0.20, 0.20, 0.75])
_P0R = np.array([0.20, -0.20, 0.75])
_Q0 = np.array([1.0, 0.0, 0.0, 0.0])


class _StubRobot:
    hand_anchor = {
        "left_hand": (_P0L, _Q0),
        "right_hand": (_P0R, _Q0),
    }
    torso_quat = _Q0
    data = SimpleNamespace(qpos=np.array([0.0, 0.0, 0.77, 1.0, 0.0, 0.0, 0.0]))


def _left_pos(traj, dt=0.002):
    n = int(round(traj.duration / dt)) + 1
    return np.array([traj.hand(i * dt)["left_hand"][0] for i in range(n)])


class TrajectoryTest(unittest.TestCase):
    def setUp(self):
        self.robot = _StubRobot()

    def test_starts_and_ends_at_anchor(self):
        for key in ("demo", "wave", "fast10"):
            traj = BimanualTrajectory(self.robot, get_scenario(key))
            np.testing.assert_allclose(traj.hand(0.0)["left_hand"][0], _P0L, atol=1e-9)
            np.testing.assert_allclose(
                traj.hand(traj.duration + 5.0)["left_hand"][0], _P0L, atol=1e-9)

    def test_right_hand_is_mirrored_when_unspecified(self):
        traj = BimanualTrajectory(self.robot, get_scenario("wave"))
        for t in (0.0, traj.duration * 0.3, traj.duration * 0.7, traj.duration):
            left = traj.hand(t)["left_hand"][0]
            right = traj.hand(t)["right_hand"][0]
            self.assertAlmostEqual(right[1], -left[1], places=9)
            self.assertAlmostEqual(right[0], left[0], places=9)
            self.assertAlmostEqual(right[2], left[2], places=9)

    def test_fast_tier_peak_speeds(self):
        """fast06/08/10 在三次样条下的实际峰值手速。

        场景的 segment 时长最初按 min-jerk 的 1.875·Δp/T 设计（标称 0.6/0.8/1.0）；
        为保证折返点速度连续改用 cubic 后，峰值约为 nominal 的 0.7 倍；
        当前路径 |Δp| = 0.230 m，峰值 0.42/0.56/0.70 m/s。
        """
        expected = {"fast06": 0.42, "fast08": 0.56, "fast10": 0.70}
        dt = 0.002
        for key, target in expected.items():
            traj = BimanualTrajectory(self.robot, get_scenario(key))
            pos = _left_pos(traj, dt)
            peak = np.linalg.norm(np.diff(pos, axis=0), axis=1).max() / dt
            self.assertAlmostEqual(peak, target, delta=0.03,
                                   msg=f"{key} 峰值手速 {peak:.3f} m/s")

    def test_fast_tiers_use_cubic(self):
        """高速档必须使用 C2 三次样条，避免每个折返点速度归零的停-走冲击。"""
        for key in ("fast06", "fast08", "fast10", "fast10_dir"):
            self.assertEqual(get_scenario(key).interp, "cubic", key)

    def test_position_is_continuous(self):
        """相邻采样点的位移不应超过峰值速度允许的范围（无跳变）。"""
        dt = 0.002
        traj = BimanualTrajectory(self.robot, get_scenario("fast10"))
        pos = _left_pos(traj, dt)
        step = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        self.assertLess(step.max() / dt, 1.2)

    def test_minjerk_has_zero_endpoint_velocity(self):
        """MinJerkSpline 自身仍保持端点/关键帧速度为零（供需要停顿的场景使用）。"""
        spline = MinJerkSpline(
            [0.0, 1.0, 2.0],
            np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        )
        for t0 in (0.0, 1.0, 2.0):
            dt = 1e-3
            p_a = spline(t0)
            p_b = spline(min(t0 + dt, spline.times[-1]))
            speed = np.linalg.norm(p_b - p_a) / dt
            self.assertLess(speed, 1e-2, f"t={t0}")


if __name__ == "__main__":
    unittest.main()
