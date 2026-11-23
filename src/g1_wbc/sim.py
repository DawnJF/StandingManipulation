"""全身 QP 规划 + 位置伺服回放。"""

import time
from dataclasses import dataclass

import mujoco
import numpy as np

from . import config
from .model import RobotModel
from .scenarios import Scenario, get_scenario
from .trajectory import BimanualTrajectory
from .utils import polygon_centroid, torso_pitch_deg
from .wbc import WholeBodyQP


@dataclass
class Plan:
    times: np.ndarray       # (N,)
    qpos: np.ndarray        # (N, 7 + 29) 基座位姿 + 身体关节
    base_z: np.ndarray      # (N,)
    com: np.ndarray         # (N, 3)
    hand_targets: list      # list[dict]
    solve_time: np.ndarray  # (N,)
    status: list


class DemoPipeline:
    def __init__(self, scenario="demo"):
        self.spec = get_scenario(scenario) if isinstance(scenario, str) else scenario
        self.robot = RobotModel(leg_gain_scale=self.spec.leg_stiffness_scale)
        self.traj = BimanualTrajectory(self.robot, self.spec)
        self.support_poly = self.robot.support_polygon()
        self.com_target = polygon_centroid(self.support_poly) + np.array(
            [config.COM_BIAS_X + self.spec.com_bias_x, 0.0])
        self.dt = config.SIM_TIMESTEP
        self.pre = self.spec.pre_hold
        self.post = self.spec.post_hold
        self.total = self.pre + self.traj.duration + self.post
        self.n_steps = int(round(self.total / self.dt))

        m = self.robot.model
        self.mocap_ids = {}
        for key, name in (("left_hand", "left_target"), ("right_hand", "right_target")):
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            self.mocap_ids[key] = int(m.body_mocapid[bid])

    # ------------------------------------------------------------------ #
    def t_traj(self, step):
        return float(np.clip(step * self.dt - self.pre, 0.0, self.traj.duration))

    def plan(self, verbose=True):
        r = self.robot
        m = r.model
        q0 = r.data.qpos.copy()
        dp = mujoco.MjData(m)
        dp.qpos[:] = q0
        dp.qvel[:] = 0.0
        mujoco.mj_forward(m, dp)

        qp = WholeBodyQP(r, dp)
        posture = dp.qpos[r.body_qpos_adr].copy()
        n = self.n_steps
        plan = Plan(
            times=np.arange(n) * self.dt,
            qpos=np.zeros((n, 7 + 29)),
            base_z=np.zeros(n),
            com=np.zeros((n, 3)),
            hand_targets=[],
            solve_time=np.zeros(n),
            status=[],
        )

        t0 = time.perf_counter()
        for step in range(n):
            tt = self.t_traj(step)
            hands = self.traj.hand(tt)
            bz = self.traj.base_z(tt)
            dp.qvel[:] = 0.0
            mujoco.mj_forward(m, dp)
            dq = qp.solve(hands, bz, self.com_target, self.support_poly, posture,
                          support_lookahead=self.spec.support_lookahead,
                          arm_vel_limit=self.spec.arm_vel_limit,
                          base_vel_limit=self.spec.base_vel_limit,
                          base_angvel_limit=self.spec.base_angvel_limit,
                          torso_quat_target=self.traj.torso_quat(tt))
            full = np.zeros(m.nv)
            full[qp.cols] = dq
            mujoco.mj_integratePos(m, dp.qpos, full, 1.0)  # full 已是单步增量

            plan.qpos[step, :7] = dp.qpos[:7]
            plan.qpos[step, 7:] = dp.qpos[r.body_qpos_adr]
            plan.base_z[step] = dp.qpos[2]
            plan.com[step] = dp.subtree_com[0]
            plan.hand_targets.append(hands)
            plan.solve_time[step] = qp.solve_time
            plan.status.append(qp.last_status)

            if verbose and step % 1000 == 0:
                err = max(np.linalg.norm(hands[k][0] - dp.site_xpos[r.site_ids[k]])
                          for k in ("left_hand", "right_hand"))
                print(f"  [plan] t={step*self.dt:5.2f}s  hand_err={err:.3f}m  "
                      f"qp={qp.solve_time*1e3:.2f}ms  {qp.last_status}")

        if verbose:
            print(f"  [plan] 完成 {n} 步，耗时 {time.perf_counter()-t0:.1f}s")
        return plan

    # ------------------------------------------------------------------ #
    def _foot_checks(self, plan):
        r, m = self.robot, self.robot.model
        dp = mujoco.MjData(m)
        drift, margins, herr = [], [], []
        for i in range(len(plan.times)):
            dp.qpos[:7] = plan.qpos[i, :7]
            dp.qpos[r.body_qpos_adr] = plan.qpos[i, 7:]
            mujoco.mj_forward(m, dp)
            drift.append(max(np.linalg.norm(dp.site_xpos[r.site_ids[k]] - r.foot_anchor[k][0])
                             for k in ("left_foot", "right_foot")))
            margins.append(r.support_margin(dp.subtree_com[0][:2], self.support_poly))
            herr.append(max(np.linalg.norm(plan.hand_targets[i][k][0]
                                           - dp.site_xpos[r.site_ids[k]])
                            for k in ("left_hand", "right_hand")))
        self.plan_hand_err = np.array(herr)
        return np.array(drift), np.array(margins)

    # ------------------------------------------------------------------ #
    def feedforward(self, plan):
        """按规划轨迹预计算重力/偏置力矩前馈（换算成位置指令偏移）。"""
        r = self.robot
        m = r.model
        n = self.n_steps
        dp = mujoco.MjData(m)
        kp = m.actuator_gainprm[r.body_act_ids, 0].copy()
        tau_ff = np.zeros((n, len(r.body_act_ids)))
        for i in range(n):
            dp.qpos[:7] = plan.qpos[i, :7]
            dp.qpos[r.body_qpos_adr] = plan.qpos[i, 7:]
            dp.qvel[:] = 0.0
            mujoco.mj_forward(m, dp)
            tau_ff[i] = dp.qfrc_bias[r.body_dof_adr]
        return np.clip(tau_ff / kp, -config.REPLAY_FF_LIMIT, config.REPLAY_FF_LIMIT)

    def set_camera(self, cam):
        """demo 视频与实时 viewer 共用的默认视角。"""
        cam.lookat[:] = [0.22, 0.0, 0.95]
        cam.distance = 2.6
        cam.azimuth = 125
        cam.elevation = -12

    # ------------------------------------------------------------------ #
    def replay(self, plan, render=False, video_path=None, verbose=True):
        r = self.robot
        m, d = r.model, r.data
        n = self.n_steps

        off_ff = self.feedforward(plan)

        # 从规划起点开始回放（与规划初始状态严格一致）
        d.qpos[:7] = plan.qpos[0, :7]
        d.qpos[r.body_qpos_adr] = plan.qpos[0, 7:]
        d.qvel[:] = 0.0
        d.ctrl[r.finger_act_ids] = r.finger_ctrl
        mujoco.mj_forward(m, d)

        writer = None
        renderer = None
        cam = None
        if render:
            renderer = mujoco.Renderer(m, height=720, width=1280)
            cam = mujoco.MjvCamera()
            self.set_camera(cam)
        if video_path is not None:
            import imageio.v2 as imageio
            writer = imageio.get_writer(str(video_path), fps=config.VIDEO_FPS,
                                        codec="libx264", quality=8, macro_block_size=1)

        log = {k: [] for k in (
            "t", "hand_pos_t", "hand_quat_t", "hand_pos", "hand_quat",
            "foot_pos", "foot_quat", "base_pos", "qpos", "qvel", "ctrl",
            "q_ref", "com", "ncon", "margin", "ff",
            "torso_pitch", "torso_pitch_t")}

        render_every = max(1, int(round(1.0 / (config.VIDEO_FPS * self.dt))))
        for step in range(n):
            tt = self.t_traj(step)
            hands = self.traj.hand(tt)
            for key in ("left_hand", "right_hand"):
                mid = self.mocap_ids[key]
                d.mocap_pos[mid] = hands[key][0]
                d.mocap_quat[mid] = hands[key][1]

            q_ref = plan.qpos[step, 7:]
            ramp = min(1.0, step * self.dt / 0.3)
            d.ctrl[r.body_act_ids] = (q_ref + ramp * off_ff[step]
                                      + self.spec.replay_k * (q_ref - d.qpos[r.body_qpos_adr]))
            d.ctrl[r.finger_act_ids] = r.finger_ctrl

            log["t"].append(step * self.dt)
            log["hand_pos_t"].append(np.concatenate([hands["left_hand"][0], hands["right_hand"][0]]))
            log["hand_quat_t"].append(np.concatenate([hands["left_hand"][1], hands["right_hand"][1]]))
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
            log["margin"].append(r.support_margin(d.subtree_com[0][:2], self.support_poly))
            log["ff"].append(off_ff[step].copy())
            log["torso_pitch"].append(torso_pitch_deg(r, d))
            log["torso_pitch_t"].append(
                float(np.degrees(self.traj.torso_pitch(tt)[0])))

            if renderer is not None and step % render_every == 0:
                renderer.update_scene(d, camera=cam)
                frame = renderer.render()
                if writer is not None:
                    writer.append_data(frame)

            mujoco.mj_step(m, d)

            if verbose and step % 1000 == 0:
                pe = max(np.linalg.norm(log["hand_pos"][-1][3 * i:3 * i + 3]
                                        - log["hand_pos_t"][-1][3 * i:3 * i + 3])
                         for i in range(2))
                print(f"  [replay] t={step*self.dt:5.2f}s  hand_err={pe:.3f}m  "
                      f"margin={log['margin'][-1]:+.3f}  ncon={d.ncon}")

        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()
        for k in log:
            log[k] = np.asarray(log[k])
        return log

    # ------------------------------------------------------------------ #
    def make_gif(self, video_path, gif_path, fps=25, scale=2):
        import imageio.v2 as imageio
        reader = imageio.get_reader(str(video_path))
        writer = imageio.get_writer(str(gif_path), mode="I", fps=fps)
        try:
            for i, frame in enumerate(reader):
                if i % 2 == 0:
                    writer.append_data(frame[::scale, ::scale])
        finally:
            writer.close()
            reader.close()
