"""实时闭环回归：站立稳定 + 短动作不摔。

覆盖此前“实时版站不住”的问题：虚拟参考状态必须与真实状态同源、全量积分。
"""

import sys
import unittest
from dataclasses import replace
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from g1_wbc import realtime
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


class RealtimeTest(unittest.TestCase):
    def setUp(self):
        self.pipe = DemoPipeline(_short_spec())
        self.q0 = self.pipe.robot.data.qpos.copy()

    def _reset(self):
        d = self.pipe.robot.data
        d.qpos[:] = self.q0
        d.qvel[:] = 0.0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(self.pipe.robot.model, d)

    def _anchors(self):
        return [self.pipe.robot.foot_anchor["left_foot"],
                self.pipe.robot.foot_anchor["right_foot"]]

    def test_standing_is_stable(self):
        self._reset()
        realtime.init_markers(self.pipe)
        res = realtime.run(
            self.pipe, realtime.mocap_target(self.pipe),
            realtime.standing_base_z(self.pipe), n_steps=1000,
            viewer=None, pace=False, verbose=False, drive_markers=False,
            target_omega=6.0, target_speed=0.4, margin_guard=0.012,
            lean_gain=-0.22, com_margin=0.015, max_target_acc=3.0)
        s = realtime.summarize(res, self._anchors())
        self.assertFalse(s["fell"])
        self.assertGreater(s["margin_min_mm"], 0.0)
        self.assertLess(s["foot_drift_max_mm"], 2.0)
        self.assertEqual(s["qp_failures"], 0)

    def test_foreaft_sway_is_damped(self):
        """0.5 Hz 双手前后摆动接近倒立摆固有频率，CoM 速度阻尼必须压住它。"""
        self._reset()
        realtime.init_markers(self.pipe)
        d = self.pipe.robot.data
        base = {k: d.mocap_pos[self.pipe.mocap_ids[k]].copy()
                for k in ("left_hand", "right_hand")}
        mocap = realtime.mocap_target(self.pipe)

        def target(step, t):
            s = min(1.0, max(0.0, (t - 0.5) / 1.0))     # 1 s 平滑进入
            w = s * s * s * (10 + s * (-15 + 6 * s))
            tau = max(0.0, t - 1.5)
            dl = np.array([0.12 * np.sin(np.pi * tau),
                           0.10 * np.cos(np.pi * tau), 0.0])   # 0.5 Hz
            dr = dl * np.array([1.0, -1.0, 1.0])
            d.mocap_pos[self.pipe.mocap_ids["left_hand"]] = base["left_hand"] + w * dl
            d.mocap_pos[self.pipe.mocap_ids["right_hand"]] = base["right_hand"] + w * dr
            return mocap(step, t)

        res = realtime.run(
            self.pipe, target, realtime.standing_base_z(self.pipe),
            n_steps=3000, viewer=None, pace=False, verbose=False,
            drive_markers=False, target_omega=6.0, target_speed=0.4,
            margin_guard=0.012, lean_gain=-0.22, com_margin=0.015,
            max_target_acc=0.8, predictive_guard=True,
            com_vel_gain=2.5, com_vel_tau=0.10,
            target_bounds=((-0.15, 0.70), (-0.50, 0.50), (0.40, 1.35)),
            bounds_rel=((-0.10, 0.12), (-0.14, 0.14), (-0.30, 0.12)),
            log_every=5)
        s = realtime.summarize(res, self._anchors())
        self.assertFalse(s["fell"])
        self.assertGreater(s["margin_min_mm"], 0.0)
        self.assertLess(s["foot_drift_max_mm"], 5.0)
        self.assertEqual(s["qp_failures"], 0)

    def test_short_motion_tracks_without_falling(self):
        self._reset()
        res = realtime.run(
            self.pipe, realtime.trajectory_target(self.pipe),
            realtime.trajectory_base_z(self.pipe), n_steps=self.pipe.n_steps,
            viewer=None, pace=False, verbose=False, drive_markers=True)
        s = realtime.summarize(res, self._anchors())
        self.assertFalse(s["fell"])
        self.assertGreater(s["margin_min_mm"], -15.0)   # 内缩边界容差
        self.assertLess(s["foot_drift_max_mm"], 5.0)
        self.assertLess(s["hand_pos_rmse_mm"], 100.0)
        self.assertEqual(s["qp_failures"], 0)


class BalanceControllerTest(unittest.TestCase):
    def test_take01_stands_at_source_speed(self):
        """原速 take01：旧运动学控制器在 2.85 s 摔倒；balance 必须原速完成。"""
        pipe = DemoPipeline("take01")
        res = realtime.run(
            pipe, realtime.trajectory_target(pipe),
            realtime.trajectory_base_z(pipe), n_steps=pipe.n_steps,
            viewer=None, pace=False, verbose=False, drive_markers=False,
            controller="balance", log_every=5)
        s = realtime.summarize(
            res, [pipe.robot.foot_anchor["left_foot"],
                  pipe.robot.foot_anchor["right_foot"]])
        self.assertFalse(s["fell"])
        self.assertGreaterEqual(s["duration_s"],
                                pipe.spec.total_duration - 0.1)
        self.assertGreater(s["margin_min_mm"], 0.0)
        self.assertLess(s["foot_drift_max_mm"], 15.0)
        self.assertLess(s["hand_pos_rmse_mm"], 50.0)
        self.assertEqual(s["qp_failures"], 0)


class WbcControllerTest(unittest.TestCase):
    def test_wave_stands_with_stable_head(self):
        """动力学 WBC：wave 原速完成，且头部前后晃动要小。"""
        pipe = DemoPipeline("wave")
        plan = pipe.plan(verbose=False)
        pipe.plan_result = plan
        pipe.plan_head = realtime.plan_head_positions(pipe, plan)
        res = realtime.run(
            pipe, realtime.trajectory_target(pipe),
            realtime.trajectory_base_z(pipe), n_steps=pipe.n_steps,
            viewer=None, pace=False, verbose=False, drive_markers=False,
            controller="wbc", log_every=1)
        s = realtime.summarize(
            res, [pipe.robot.foot_anchor["left_foot"],
                  pipe.robot.foot_anchor["right_foot"]])
        self.assertFalse(s["fell"])
        self.assertGreater(s["margin_min_mm"], 0.0)
        self.assertLess(s["foot_drift_max_mm"], 15.0)
        self.assertLess(s["hand_pos_rmse_mm"], 20.0)
        self.assertLess(s["hand_rot_rmse_deg"], 5.0)
        self.assertEqual(s["qp_failures"], 0)
        head_x = np.asarray(res["log"]["head_pos"])[:, 0]
        dt = float(res["log"]["t"][1] - res["log"]["t"][0])
        self.assertLess((head_x.max() - head_x.min()) * 1e3, 60.0)
        self.assertLess(np.sqrt(np.mean(np.gradient(head_x, dt) ** 2)), 0.09)


if __name__ == "__main__":
    unittest.main()
