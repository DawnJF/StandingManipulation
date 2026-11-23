"""端到端冒烟测试：加载模型 -> 规划 -> 回放（不渲染）。

用一个 1 s 的极简场景，检查路径/资源引用、QP 可行性、双脚不滑动与不摔倒。
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g1_wbc import config
from g1_wbc.metrics import compute_metrics
from g1_wbc.scenarios import get_scenario
from g1_wbc.sim import DemoPipeline


def _short_spec():
    base = get_scenario("demo")
    return replace(
        base,
        times=(0.0, 0.6),
        time_scale=1.0,
        pre_hold=0.2,
        post_hold=0.2,
        interp="minjerk",
        left_pos=(None, (0.27, 0.20, 0.95)),
        left_rpy_deg=((0, 0, 0), (0, -80, 0)),
        right_pos=None,
        right_rpy_deg=None,
        base_dz=(0.0, -0.010),
    )


class AssetPathTest(unittest.TestCase):
    def test_scene_xml_exists(self):
        self.assertTrue(config.SCENE_XML.is_file(), config.SCENE_XML)
        self.assertTrue((config.ASSET_DIR / "meshes").is_dir())


class PipelineSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = DemoPipeline(_short_spec())
        cls.plan = cls.pipe.plan(verbose=False)
        cls.foot_plan, cls.margin_plan = cls.pipe._foot_checks(cls.plan)
        cls.log = cls.pipe.replay(cls.plan, render=False, video_path=None,
                                  verbose=False)
        cls.res = compute_metrics(cls.pipe, cls.plan, cls.log,
                                  cls.foot_plan, cls.margin_plan)

    def test_all_qp_steps_solved(self):
        self.assertEqual(sum(1 for s in self.plan.status if s != "solved"), 0)

    def test_feet_do_not_slide_in_plan(self):
        self.assertLess(self.foot_plan.max() * 1e3, 0.5)   # mm

    def test_plan_keeps_com_inside_support(self):
        self.assertGreater(self.margin_plan.min(), 0.0)

    def test_plan_tracks_targets(self):
        self.assertLess(self.res["plan"]["hand_pos_err_mean_mm"], 60.0)

    def test_replay_does_not_fall(self):
        self.assertFalse(self.res["balance"]["fell"])
        self.assertLess(self.res["balance"]["foot_drift_max_mm"], 5.0)


if __name__ == "__main__":
    unittest.main()
