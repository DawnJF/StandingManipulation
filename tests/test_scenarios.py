"""场景定义的结构性检查（不需要 MuJoCo）。"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g1_wbc.scenarios import SCENARIOS, describe_scenarios, get_scenario


class ScenarioSpecTest(unittest.TestCase):
    def test_arrays_match_times(self):
        for key, spec in SCENARIOS.items():
            n = len(spec.times)
            self.assertGreaterEqual(n, 2, f"{key} 至少需要 2 个关键帧")
            self.assertEqual(len(spec.left_pos), n, f"{key}: left_pos")
            self.assertEqual(len(spec.left_rpy_deg), n, f"{key}: left_rpy_deg")
            if spec.right_pos is not None:
                self.assertEqual(len(spec.right_pos), n, f"{key}: right_pos")
            if spec.right_rpy_deg is not None:
                self.assertEqual(len(spec.right_rpy_deg), n, f"{key}: right_rpy_deg")
            if spec.base_dz:
                self.assertEqual(len(spec.base_dz), n, f"{key}: base_dz")

    def test_times_increasing(self):
        for key, spec in SCENARIOS.items():
            diffs = [b - a for a, b in zip(spec.times, spec.times[1:])]
            self.assertTrue(all(d > 0 for d in diffs), f"{key}: times 必须严格递增")

    def test_takes_keep_source_speed(self):
        """动捕数据必须保留原始 FBX 时序，禁止用时间缩放降低评估难度。"""
        for key in ("take01", "take02", "take03"):
            spec = get_scenario(key)
            self.assertEqual(spec.time_scale, 1.0,
                             f"{key}: 不得对原始动捕做时间缩放")
            self.assertAlmostEqual(spec.duration, spec.times[-1], places=9,
                                   msg=f"{key}: 动作时长必须等于源时长")

    def test_positive_durations_and_limits(self):
        for key, spec in SCENARIOS.items():
            self.assertGreater(spec.time_scale, 0, key)
            self.assertGreater(spec.duration, 0, key)
            self.assertGreater(spec.pre_hold, 0, key)
            self.assertGreater(spec.post_hold, 0, key)
            self.assertGreater(spec.arm_vel_limit, 0, key)
            self.assertGreater(spec.base_vel_limit, 0, key)
            self.assertGreater(spec.base_angvel_limit, 0, key)
            self.assertGreater(spec.replay_k, 0, key)
            self.assertIn(spec.interp, ("cubic", "minjerk"), key)

    def test_get_scenario(self):
        self.assertEqual(get_scenario("wave").key, "wave")
        with self.assertRaises(KeyError):
            get_scenario("not-a-scenario")

    def test_fast_tiers_faster_than_existing(self):
        """fast06/08/10 的单段时长应依次变短（同一几何路径）。"""
        t = [SCENARIOS[k].times for k in ("fast06", "fast08", "fast10")]
        seg = [times[2] - times[1] for times in t]
        self.assertGreater(seg[0], seg[1])
        self.assertGreater(seg[1], seg[2])

    def test_describe_scenarios_lists_all(self):
        text = describe_scenarios()
        for key in SCENARIOS:
            self.assertIn(key, text)


if __name__ == "__main__":
    unittest.main()
