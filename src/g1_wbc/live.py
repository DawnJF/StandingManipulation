"""实时闭环可视化：在线求解双手 6-DOF 目标，MuJoCo viewer 实时显示。

用法（macOS 必须用 mjpython 启动窗口）：
    mjpython -m g1_wbc.live --scenario demo                  # 跟踪场景轨迹
    mjpython -m g1_wbc.live --scenario demo --target mocap   # 拖动 mocap 目标
    mjpython -m g1_wbc.live --scenario demo --loop
    python    -m g1_wbc.live --list                          # 列出全部场景
    python    -m g1_wbc.live --scenario demo --no-viewer     # 只跑数值
"""

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco

from . import realtime
from .scenarios import SCENARIOS, describe_scenarios, get_scenario
from .sim import DemoPipeline


def _ensure_viewer_runtime():
    """macOS 上把当前进程切换为 mjpython（Cocoa 事件循环）。

    若系统 Xcode 许可未接受，/usr/bin/otool 会报错，mjpython 无法启动；
    优先使用 CommandLineTools 自带的 otool。返回 False 表示找不到 mjpython。
    """
    if sys.platform != "darwin":
        return True
    import mujoco.viewer
    if getattr(mujoco.viewer, "_MJPYTHON", None) is not None:
        return True

    mjpython = Path(sys.executable).with_name("mjpython")
    if not mjpython.exists():
        return False
    env = dict(os.environ)
    clt_bin = Path("/Library/Developer/CommandLineTools/usr/bin")
    if (clt_bin / "otool").exists():
        env["PATH"] = str(clt_bin) + os.pathsep + env.get("PATH", "")
    print("正在通过 mjpython 启动交互窗口 ...", flush=True)
    os.execvpe(str(mjpython), [str(mjpython), "-m", "g1_wbc.live", *sys.argv[1:]], env)
    return True


def _print_rt_summary(pipe, res):
    s = realtime.summarize(
        res, [pipe.robot.foot_anchor["left_foot"],
              pipe.robot.foot_anchor["right_foot"]])
    if not s.get("steps"):
        print("[实时] 未运行任何步。", flush=True)
        return
    print(f"[实时] {s['duration_s']:.1f} s  base_z min={s['base_z_min']:.3f}  "
          f"CoM 裕度 min={s['margin_min_mm']:+.1f} mm  "
          f"位置 RMSE={s['hand_pos_rmse_mm']:.1f} mm  "
          f"足端漂移 max={s.get('foot_drift_max_mm', 0.0):.2f} mm  "
          f"QP 失败={s['qp_failures']}  hold={s['qp_holds']}  "
          f"{'摔倒' if s['fell'] else '站立完成'}", flush=True)


def run_plan(args):
    """离线 IK 规划回放：只显示 plan() 的关节轨迹，不跑物理、不跟踪。"""
    pipe = DemoPipeline(args.scenario)
    r = pipe.robot
    d = r.data
    print(f"[IK] 规划 {args.scenario} ...", flush=True)
    plan = pipe.plan(verbose=True)
    foot, margin = pipe._foot_checks(plan)
    print(f"[IK] 足端漂移 max={foot.max()*1e3:.2f} mm  "
          f"CoM 裕度 min={margin.min()*1e3:+.1f} mm  "
          f"QP 失败={sum(1 for s in plan.status if s != 'solved')}  "
          f"手位置误差 max={pipe.plan_hand_err.max()*1e3:.1f} mm", flush=True)
    if args.no_viewer:
        return

    dt = pipe.dt
    with mujoco.viewer.launch_passive(r.model, d) as viewer:
        pipe.set_camera(viewer.cam)
        print("IK 规划回放：关闭窗口退出；拖动鼠标可旋转视角。", flush=True)
        while viewer.is_running():
            for i in range(len(plan.times)):
                if not viewer.is_running():
                    break
                d.qpos[:7] = plan.qpos[i, :7]
                d.qpos[r.body_qpos_adr] = plan.qpos[i, 7:]
                d.qvel[:] = 0.0
                mujoco.mj_forward(r.model, d)
                for key, target in plan.hand_targets[i].items():
                    mid = pipe.mocap_ids[key]
                    d.mocap_pos[mid] = target[0]
                    d.mocap_quat[mid] = target[1]
                viewer.sync()
                time.sleep(max(dt / max(args.speed, 1e-3), 0.0))
            if not args.loop:
                while viewer.is_running():
                    time.sleep(0.05)
                break


def run_realtime(args):
    """在线闭环模式：不预规划，直接按目标实时求解。"""
    spec = get_scenario(args.scenario)
    print(f"[场景] {spec.key} — {spec.title}\n  {spec.description}\n"
          f"  轨迹 {spec.duration:.1f} s（含前后静止共 {spec.total_duration:.1f} s）",
          flush=True)

    pipe = DemoPipeline(args.scenario)
    r = pipe.robot
    d = r.data
    q0 = d.qpos.copy()

    if args.target == "mocap":
        realtime.init_markers(pipe)
        target_fn = realtime.mocap_target(pipe)
        base_z_fn = realtime.standing_base_z(pipe)
        n_steps = 10 ** 9
        shape = dict(target_omega=6.0, target_speed=0.40, margin_guard=0.012,
                     lean_gain=-0.22, com_margin=0.015, max_target_acc=0.8,
                     predictive_guard=True, com_vel_gain=2.5, com_vel_tau=0.10,
                     target_bounds=((-0.15, 0.70), (-0.50, 0.50), (0.40, 1.35)),
                     bounds_rel=((-0.10, 0.12), (-0.14, 0.14), (-0.30, 0.12)),
                     log_every=5)          # 100 Hz 记录，避免长时间拖动占满内存
        drive_markers = False
    else:
        target_fn = realtime.trajectory_target(pipe)
        base_z_fn = realtime.trajectory_base_z(pipe)
        n_steps = pipe.n_steps
        shape = dict(target_omega=100.0, target_speed=None, margin_guard=0.0,
                     lean_gain=0.0)
        drive_markers = True

    if args.no_viewer:
        steps = 2500 if args.target == "mocap" else n_steps
        res = realtime.run(pipe, target_fn, base_z_fn, n_steps=steps,
                           viewer=None, pace=False, verbose=True,
                           drive_markers=drive_markers,
                           controller=args.controller, **shape)
        _print_rt_summary(pipe, res)
        return

    with mujoco.viewer.launch_passive(r.model, d) as viewer:
        pipe.set_camera(viewer.cam)
        print("窗口已打开：鼠标拖拽旋转、滚轮缩放；关闭窗口即退出。", flush=True)
        if args.target == "mocap":
            print("拖动两个半透明标记即可实时改变手部目标。", flush=True)
        while viewer.is_running():
            d.qpos[:] = q0
            d.qvel[:] = 0.0
            d.ctrl[:] = 0.0
            mujoco.mj_forward(r.model, d)
            res = realtime.run(pipe, target_fn, base_z_fn, n_steps=n_steps,
                               viewer=viewer, speed=args.speed, pace=True,
                               verbose=True, drive_markers=drive_markers,
                               controller=args.controller, **shape)
            _print_rt_summary(pipe, res)

            if args.target == "mocap" or not args.loop:
                while viewer.is_running():
                    time.sleep(0.05)
                break
            print("循环播放中 ...", flush=True)


def main():
    ap = argparse.ArgumentParser(description="G1 双手 6-DOF 实时闭环可视化")
    ap.add_argument("--scenario", default="demo", choices=list(SCENARIOS))
    ap.add_argument("--target", choices=("trajectory", "mocap"), default="trajectory",
                    help="目标源：场景轨迹或 mocap 标记（可拖动）")
    ap.add_argument("--loop", action="store_true", help="循环播放")
    ap.add_argument("--speed", type=float, default=1.0, help="回放倍速（默认 1.0）")
    ap.add_argument("--no-viewer", action="store_true", help="只跑数值并打印摘要")
    ap.add_argument("--controller", choices=("kinematic", "balance", "wbc"),
                    default="wbc",
                    help="wbc=动力学 WBC（默认，原速 16/16）；"
                         "balance=运动学 QP + 平衡前馈（对照）；"
                         "kinematic=位置伺服回放")
    ap.add_argument("--plan", action="store_true",
                    help="只回放离线 IK 规划轨迹（不跑物理/控制），用于检查规划本身")
    ap.add_argument("--list", action="store_true", help="列出全部场景后退出")
    args = ap.parse_args()

    if args.list:
        print(describe_scenarios())
        return

    if not args.no_viewer and not _ensure_viewer_runtime():
        print("未找到 mjpython（应为与 python 同目录的可执行文件）。\n"
              "请确认在 conda 环境内，并重新安装 mujoco。", file=sys.stderr)
        sys.exit(1)

    try:
        if args.plan:
            run_plan(args)
        else:
            run_realtime(args)
    except KeyboardInterrupt:
        print("\n已退出。")


if __name__ == "__main__":
    main()
