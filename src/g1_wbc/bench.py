"""无渲染评估全部场景：追踪误差、平衡与平滑性指标。

用法：
    python -m g1_wbc.bench                 # 评估全部场景
    python -m g1_wbc.bench --scenario wave # 只评估一个场景
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from . import config, realtime
from .kin_check import stats_from_qpos
from .metrics import compute_metrics
from .scenarios import SCENARIOS
from .sim import DemoPipeline

RESULT_DIR = config.PKG_ROOT / "results"


def evaluate(key, controller="wbc"):
    pipe = DemoPipeline(key)
    plan = pipe.plan(verbose=False)
    foot_plan, margin_plan = pipe._foot_checks(plan)
    # 头部参考轨迹：WBC 用作头部稳定任务的目标，metrics 用于统计额外晃动
    pipe.plan_result = plan
    pipe.plan_head = realtime.plan_head_positions(pipe, plan)

    # 使用与 live/viz 完全一致的实时控制器（含自适应调速），而不是离线开环回放；
    # 否则 bench 会报告已被在线调速避免的摔倒。
    r = pipe.robot
    d = r.data
    d.qpos[:7] = plan.qpos[0, :7]
    d.qpos[r.body_qpos_adr] = plan.qpos[0, 7:]
    d.qvel[:] = 0.0
    d.ctrl[:] = 0.0
    mujoco.mj_forward(r.model, d)
    run = realtime.run(pipe, realtime.trajectory_target(pipe),
                       realtime.trajectory_base_z(pipe), n_steps=pipe.n_steps,
                       viewer=None, pace=False, verbose=False,
                       drive_markers=False, controller=controller)
    log = run["log"]
    res = compute_metrics(pipe, plan, log, foot_plan, margin_plan,
                          plan_head=pipe.plan_head)

    # 目标轨迹自身的峰值手速（设计值）与实测峰值
    pos = np.array([[pipe.traj.hand(pipe.t_traj(s))[k][0]
                     for k in ("left_hand", "right_hand")]
                    for s in range(pipe.n_steps)])
    hp = log["hand_pos"]
    v_left = np.linalg.norm(np.gradient(hp[:, :3], log["t"], axis=0), axis=1)
    v_right = np.linalg.norm(np.gradient(hp[:, 3:6], log["t"], axis=0), axis=1)
    # 峰值只统计摔倒前，避免倒地后的地面运动污染手速指标
    t_end = res["balance"]["fall_time_s"]
    if t_end is not None:
        m = log["t"] <= t_end
        v_left, v_right = v_left[m], v_right[m]
    res["scenario"] = key
    res["title"] = pipe.spec.title
    res["controller"] = controller
    res["target_peak_speed_mps"] = float(
        np.linalg.norm(np.diff(pos, axis=0), axis=2).max() / pipe.dt)
    res["actual_peak_speed_mps"] = float(max(v_left.max(), v_right.max()))
    res["actual_peak_left_mps"] = float(v_left.max())
    res["actual_peak_right_mps"] = float(v_right.max())
    res["realtime"] = {
        "wall_time_s": float(run["wall_time"]),
        "realtime_factor": float(run["steps"] * pipe.dt
                                 / max(run["wall_time"], 1e-9)),
        "solve_time_mean_ms": float(np.mean(run["solve_time"]) * 1e3),
        "solve_time_p95_ms": float(np.percentile(run["solve_time"], 95) * 1e3),
        "controller_failures": int(sum(1 for s in run["status"]
                                       if s == "failed")),
        "controller_holds": int(sum(1 for s in run["status"]
                                    if s == "hold")),
    }

    # 附带记录参考轨迹上的几何穿透（不参与评估、不迭代优化）
    col = stats_from_qpos(pipe, plan.qpos, pipe.dt, stride=5)
    res["collision"] = {
        "max_depth_mm": float(col["max_depth_mm"]),
        "hit_frames": col["hit_frames"],
        "seconds": float(col["hit_frames"] * pipe.dt * 5),
        "pairs": [{"pair": list(p), "kind": v[0], "frames": v[1],
                   "depth_mm": v[2] * 1e3, "t0": v[3], "t1": v[4]}
                  for p, v in col["pairs"].items()],
    }
    return res


def _row(res):
    b, s, p = res["balance"], res["smoothness"], res["plan"]
    tp, col = res["torso_pitch"], res["collision"]
    hd = res["head"]
    status = ("ok" if not b["fell"]
              else f"FALL@{b['fall_time_s']:.2f}s")
    return (f"{res['scenario']:10s} {res['duration_s']:5.1f}s  "
            f"pos {res['hand_pos_err_mm']['rmse']:6.1f} mm  "
            f"rot {res['hand_rot_err_deg']['rmse']:5.2f}°  "
            f"margin {b['margin_hull_min_mm']:7.1f} mm  "
            f"drift {b['foot_drift_max_mm']:6.2f} mm  "
            f"head {hd['x_pp_mm']:5.1f}/{hd['err_x_rmse_mm']:4.1f} mm  "
            f"acc {s['steady_joint_acc_rms']:6.3f}  "
            f"jerk {s['steady_joint_jerk_rms']:7.2f}  "
            f"peak {res['actual_peak_speed_mps']:4.2f}/"
            f"{res['target_peak_speed_mps']:4.2f} m/s  "
            f"torso {tp['actual_peak_deg']:5.1f}/{tp['cmd_peak_deg']:5.1f}"
            f"(±{tp['rmse_deg']:.1f})  "
            f"pen {col['max_depth_mm']:5.2f}mm/{col['seconds']:.2f}s  "
            f"qp {p['failed_steps']:3d}  {status}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="all", choices=["all", *SCENARIOS])
    ap.add_argument("--controller", choices=("kinematic", "balance", "wbc"),
                    default="wbc",
                    help="wbc=动力学 WBC（默认，原速 16/16）；"
                         "balance=平衡前馈（原速对照）；kinematic=位置伺服回放")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    keys = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    out = Path(args.out) if args.out else RESULT_DIR / (
        "scenario_metrics_wbc.json" if args.controller == "wbc"
        else "scenario_metrics_balance.json" if args.controller == "balance"
        else "scenario_metrics.json")
    results = {}
    for key in keys:
        t0 = time.perf_counter()
        res = evaluate(key, controller=args.controller)
        results[key] = res
        print(_row(res) + f"  qp {res['plan']['solve_time_mean_ms']:.2f} ms"
              f"  ({time.perf_counter()-t0:.0f}s)", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"\n写入 {out}")


if __name__ == "__main__":
    main()
