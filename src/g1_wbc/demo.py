"""入口：全身 QP 规划 -> 位置伺服回放 -> 视频/GIF + 指标。

用法（在工程根目录 /Users/majianfei/Projects/StandingManipulation 下）：
    ./scripts/demo.sh              # 生成视频/GIF/指标
    ./scripts/demo.sh --no-video   # 只跑数值
"""

import argparse
import time
from pathlib import Path

import numpy as np

from . import config
from .metrics import compute_metrics, make_plots, save_json, save_timeseries
from .sim import DemoPipeline

RESULT_DIR = config.PKG_ROOT / "results"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-video", action="store_true")
    ap.add_argument("--out", default=str(RESULT_DIR))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    pipe = DemoPipeline()
    print(f"[init] 初始站姿 base={np.round(pipe.robot.data.qpos[:3], 3)}  "
          f"CoM 裕度={pipe.robot.support_margin(pipe.robot.data.subtree_com[0][:2]):+.3f} m")

    print("[1/3] 规划")
    plan = pipe.plan()
    foot_plan, margin_plan = pipe._foot_checks(plan)
    print(f"[plan] 足端漂移 max={foot_plan.max()*1e3:.3f} mm  "
          f"CoM 裕度 min={margin_plan.min():+.4f} m  "
          f"QP 失败步数={sum(1 for s in plan.status if s != 'solved')}")

    print("[2/3] 回放")
    video = None if args.no_video else out / "demo.mp4"
    log = pipe.replay(plan, render=not args.no_video, video_path=video)

    print("[3/3] 指标")
    res = compute_metrics(pipe, plan, log, foot_plan, margin_plan)
    save_json(res, out / "metrics.json")
    save_timeseries(log, out / "timeseries.csv")
    make_plots(pipe, log, out)

    if video is not None:
        gif = out / "demo.gif"
        print("  [gif] 生成中 ...")
        pipe.make_gif(video, gif)

    print("\n==== 结果 ====")
    print(f"时长 {res['duration_s']:.1f}s  步数 {res['steps']}")
    print(f"手部位置误差  RMSE {res['hand_pos_err_mm']['rmse']:.1f} mm  "
          f"mean {res['hand_pos_err_mm']['mean']:.1f} mm  "
          f"max {res['hand_pos_err_mm']['max']:.1f} mm  "
          f"(L/R RMSE {res['hand_pos_err_mm']['left_rmse']:.1f}/"
          f"{res['hand_pos_err_mm']['right_rmse']:.1f} mm)")
    print(f"手部姿态误差  RMSE {res['hand_rot_err_deg']['rmse']:.2f}°  "
          f"max {res['hand_rot_err_deg']['max']:.2f}°")
    b = res["balance"]
    print(f"平衡          CoM 裕度 min {b['margin_min_m']*1e3:.1f} mm  "
          f"足端漂移 max {b['foot_drift_max_mm']:.2f} mm  "
          f"ncon_min {b['ncon_min']}  摔倒={b['fell']}")
    s = res["smoothness"]
    print(f"平滑性        关节 |vel|max {s['joint_vel_max']:.2f} rad/s  "
          f"|acc|max {s['joint_acc_max']:.2f} rad/s²  acc RMS {s['joint_acc_rms']:.2f}  "
          f"jerk RMS {s['joint_jerk_rms']:.1f} rad/s³")
    print(f"              EE |acc|max {s['ee_acc_max']:.2f} m/s²  "
          f"EE jerk RMS {s['ee_jerk_rms']:.2f} m/s³")
    p = res["plan"]
    print(f"QP            每步 {p['solve_time_mean_ms']:.2f} ms "
          f"(p95 {p['solve_time_p95_ms']:.2f}, max {p['solve_time_max_ms']:.2f})  "
          f"失败 {p['failed_steps']}")
    print(f"\n总耗时 {time.perf_counter()-t0:.1f}s  输出目录 {out}")


if __name__ == "__main__":
    main()
