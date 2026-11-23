"""追踪误差、平滑性与平衡指标。"""

import json
from pathlib import Path

import numpy as np

from . import config
from .utils import quat_error


def _diff(x, dt, order=1):
    for _ in range(order):
        x = np.diff(x, axis=0) / dt
    return x


def _torso_metrics(log, mask):
    act = np.asarray(log.get("torso_pitch", []), float)
    cmd = np.asarray(log.get("torso_pitch_t", []), float)
    if act.size == 0 or cmd.size == 0:
        return {"cmd_peak_deg": 0.0, "actual_peak_deg": 0.0,
                "rmse_deg": 0.0, "peak_err_deg": 0.0}
    n = min(len(act), len(cmd), len(mask))
    act, cmd, m = act[:n], cmd[:n], mask[:n]
    err = act[m] - cmd[m]
    return {
        "cmd_peak_deg": float(np.max(np.abs(cmd[m]))),
        "actual_peak_deg": float(np.max(np.abs(act[m]))),
        "rmse_deg": float(np.sqrt(np.mean(err ** 2))),
        "peak_err_deg": float(np.max(np.abs(err))),
    }


def _head_metrics(log, mask, plan_head=None):
    """头部晃动：前后/侧向峰峰值、俯仰峰峰值，以及相对规划头部的偏差。"""
    pos = np.asarray(log.get("head_pos", []), float)
    quat = np.asarray(log.get("head_quat", []), float)
    if pos.size == 0:
        return {"x_pp_mm": 0.0, "y_pp_mm": 0.0, "z_pp_mm": 0.0,
                "x_vel_rms_mps": 0.0, "pitch_pp_deg": 0.0,
                "err_x_rmse_mm": 0.0, "err_x_max_mm": 0.0,
                "err_y_rmse_mm": 0.0}
    n = min(len(pos), len(quat), len(mask))
    pos, quat, m = pos[:n], quat[:n], np.asarray(mask[:n], bool)
    if not np.any(m):
        m = np.ones(n, bool)
    p, q = pos[m], quat[m]
    dt = (float(log["t"][1] - log["t"][0]) if len(log["t"]) > 1
          else config.SIM_TIMESTEP)
    vx = np.gradient(pos[:, 0], dt)[m]
    # 四元数 [w,x,y,z] 的俯仰角
    pitch = np.rad2deg(np.arcsin(np.clip(
        2.0 * (q[:, 0] * q[:, 2] - q[:, 3] * q[:, 1]), -1.0, 1.0)))
    out = {
        "x_pp_mm": float((p[:, 0].max() - p[:, 0].min()) * 1e3),
        "y_pp_mm": float((p[:, 1].max() - p[:, 1].min()) * 1e3),
        "z_pp_mm": float((p[:, 2].max() - p[:, 2].min()) * 1e3),
        "x_vel_rms_mps": float(np.sqrt(np.mean(vx ** 2))),
        "pitch_pp_deg": float(pitch.max() - pitch.min()),
        "err_x_rmse_mm": 0.0,
        "err_x_max_mm": 0.0,
        "err_y_rmse_mm": 0.0,
    }
    if plan_head is not None and len(plan_head) > 0:
        idx = np.clip(np.rint(log["t"][:n] / dt).astype(int),
                      0, len(plan_head) - 1)
        ref = np.asarray(plan_head, float)[idx][m]
        if len(ref) == len(p):
            ex = (p[:, 0] - ref[:, 0]) * 1e3
            ey = (p[:, 1] - ref[:, 1]) * 1e3
            out["err_x_rmse_mm"] = float(np.sqrt(np.mean(ex ** 2)))
            out["err_x_max_mm"] = float(np.max(np.abs(ex)))
            out["err_y_rmse_mm"] = float(np.sqrt(np.mean(ey ** 2)))
    return out


def compute_metrics(pipeline, plan, log, foot_drift_plan, margin_plan,
                    plan_head=None):
    r = pipeline.robot
    dt = pipeline.dt
    t = log["t"]
    hp_t, hp = log["hand_pos_t"], log["hand_pos"]
    hq_t, hq = log["hand_quat_t"], log["hand_quat"]

    pos_err_l = np.linalg.norm(hp[:, :3] - hp_t[:, :3], axis=1)
    pos_err_r = np.linalg.norm(hp[:, 3:] - hp_t[:, 3:], axis=1)
    # 左右两手误差的 RMS（不是把两个 3 维误差向量拼成一个 6 维向量求范数）
    pos_err = np.sqrt(0.5 * (pos_err_l ** 2 + pos_err_r ** 2))
    rot_err = np.stack([np.array([np.linalg.norm(quat_error(hq_t[i, 4 * k:4 * k + 4],
                                                          hq[i, 4 * k:4 * k + 4]))
                                  for i in range(len(t))])
                        for k in range(2)], axis=1)
    rot_err_deg = np.rad2deg(rot_err)

    foot_drift = np.linalg.norm(log["foot_pos"] - log["foot_pos"][0], axis=1)
    foot_rot = np.stack([np.array([np.linalg.norm(quat_error(log["foot_quat"][0, 4 * k:4 * k + 4],
                                                           log["foot_quat"][i, 4 * k:4 * k + 4]))
                                   for i in range(len(t))])
                         for k in range(2)], axis=1)
    foot_rot_deg = np.rad2deg(foot_rot).max(axis=1)

    body_dof = r.body_dof_adr
    qvel = log["qvel"][:, body_dof]
    qacc = _diff(qvel, dt, 1)
    qjerk = _diff(qvel, dt, 2)
    base_vel = log["qvel"][:, :3]
    ee_vel = _diff(hp, dt, 1)
    ee_acc = _diff(hp, dt, 2)
    ee_jerk = _diff(hp, dt, 3)
    err_acc = _diff(pos_err, dt, 2)
    err_jerk = _diff(pos_err, dt, 3)

    # 摔倒检测：基座高度掉到 0.45 m 以下即判定摔倒。若摔倒，所有统计截断到摔倒前，
    # 否则倒地后的巨大误差会掩盖“到底在多快时开始跟不上”的信息。
    base_z = log["base_pos"][:, 2]
    # 摔倒阈值：既要低于平站 0.77 m，也要低于场景规划的最低骨盆高度
    # （pickup 下蹲本来就到 0.45 m，不能用固定阈值误判）。
    fell_thr = min(0.45, float(np.min(plan.qpos[:, 2])) - 0.08)
    fell_idx = np.where(base_z < fell_thr)[0]
    fell = bool(len(fell_idx))
    fall_time = float(t[fell_idx[0]]) if fell else float(t[-1])
    m = (t >= pipeline.pre) & (t <= fall_time)
    if not np.any(m):
        m = t >= pipeline.pre

    # 稳态窗口同时受摔倒时刻限制（acc 数组比 t 短 1，jerk 短 2）
    i0 = max(int(0.5 / dt), 1)
    i1_acc = min(max(int(fall_time / dt), i0 + 1), len(qacc))
    i1_jerk = min(max(int(fall_time / dt), i0 + 1), len(qjerk))

    res = {
        "duration_s": float(t[-1]),
        "steps": int(len(t)),
        "hand_pos_err_mm": {
            "rmse": float(np.sqrt(np.mean(pos_err[m] ** 2)) * 1e3),
            "mean": float(np.mean(pos_err[m]) * 1e3),
            "max": float(np.max(pos_err[m]) * 1e3),
            "left_rmse": float(np.sqrt(np.mean(pos_err_l[m] ** 2)) * 1e3),
            "right_rmse": float(np.sqrt(np.mean(pos_err_r[m] ** 2)) * 1e3),
        },
        "hand_rot_err_deg": {
            "rmse": float(np.sqrt(np.mean(rot_err_deg[m] ** 2))),
            "mean": float(np.mean(rot_err_deg[m])),
            "max": float(np.max(rot_err_deg[m])),
        },
        "balance": {
            "margin_min_m": float(np.min(log["margin"][m])),
            "margin_hull_min_mm": float((np.min(log["margin"][m]) + config.SUPPORT_SHRINK
                                         + float(r.model.geom_size[r.foot_contact_geoms[0], 0])) * 1e3),
            "margin_mean_m": float(np.mean(log["margin"][m])),
            "foot_drift_max_mm": float(np.max(foot_drift[m]) * 1e3),
            "foot_rot_drift_max_deg": float(np.max(foot_rot_deg[m])),
            "ncon_min": int(np.min(log["ncon"][m])),
            "base_z_min": float(np.min(base_z)),
            "fell": fell,
            "fall_time_s": fall_time if fell else None,
        },
        "smoothness": {
            "joint_vel_max": float(np.max(np.abs(qvel))),
            "joint_vel_rms": float(np.sqrt(np.mean(qvel ** 2))),
            "joint_acc_max": float(np.max(np.abs(qacc))),
            "joint_acc_max_t": float(t[1:][int(np.argmax(np.max(np.abs(qacc), axis=1)))]),
            "joint_acc_rms": float(np.sqrt(np.mean(qacc ** 2))),
            "joint_jerk_rms": float(np.sqrt(np.mean(qjerk ** 2))),
            "joint_jerk_max": float(np.max(np.abs(qjerk))),
            "ee_acc_max": float(np.max(np.linalg.norm(ee_acc, axis=1))),
            "ee_acc_max_t": float(t[2:][int(np.argmax(np.linalg.norm(ee_acc, axis=1)))]),
            "ee_acc_rms": float(np.sqrt(np.mean(np.sum(ee_acc ** 2, axis=1)))),
            "ee_jerk_rms": float(np.sqrt(np.mean(np.sum(ee_jerk ** 2, axis=1)))),
            "err_acc_rms_mm": float(np.sqrt(np.mean(err_acc ** 2)) * 1e3),
            "err_jerk_rms_mm": float(np.sqrt(np.mean(err_jerk ** 2)) * 1e3),
            "base_vel_max": float(np.max(np.linalg.norm(base_vel, axis=1))),
            "steady_joint_acc_rms": float(np.sqrt(np.mean(
                qacc[i0:i1_acc] ** 2))),
            "steady_joint_jerk_rms": float(np.sqrt(np.mean(
                qjerk[i0:i1_jerk] ** 2))),
            "steady_ee_acc_rms": float(np.sqrt(np.mean(np.sum(
                ee_acc[i0:i1_acc] ** 2, axis=1)))),
            "steady_ee_jerk_rms": float(np.sqrt(np.mean(np.sum(
                ee_jerk[i0:i1_jerk] ** 2, axis=1)))),
        },
        "plan": {
            "solve_time_mean_ms": float(np.mean(plan.solve_time) * 1e3),
            "solve_time_max_ms": float(np.max(plan.solve_time) * 1e3),
            "solve_time_p95_ms": float(np.percentile(plan.solve_time, 95) * 1e3),
            "failed_steps": int(sum(1 for s in plan.status if s != "solved")),
            "foot_drift_max_mm": float(np.max(foot_drift_plan) * 1e3),
            "margin_min_m": float(np.min(margin_plan)),
            "hand_pos_err_mean_mm": float(np.mean(pipeline.plan_hand_err) * 1e3),
            "hand_pos_err_max_mm": float(np.max(pipeline.plan_hand_err) * 1e3),
        },
        "torso_pitch": _torso_metrics(log, m),
        "head": _head_metrics(log, m, plan_head),
    }
    return res


def save_timeseries(log, path):
    cols = {
        "t": log["t"],
        "hand_pos_err": np.sqrt(0.5 * (np.sum((log["hand_pos"][:, :3] - log["hand_pos_t"][:, :3]) ** 2, axis=1)
                                       + np.sum((log["hand_pos"][:, 3:] - log["hand_pos_t"][:, 3:]) ** 2, axis=1))),
        "margin": log["margin"],
        "foot_drift": np.linalg.norm(log["foot_pos"] - log["foot_pos"][0], axis=1),
        "ncon": log["ncon"],
        "base_x": log["base_pos"][:, 0],
        "base_y": log["base_pos"][:, 1],
        "base_z": log["base_pos"][:, 2],
    }
    keys = list(cols)
    arr = np.column_stack([cols[k] for k in keys])
    np.savetxt(path, arr, delimiter=",", header=",".join(keys), comments="")


def save_json(res, path):
    Path(path).write_text(json.dumps(res, indent=2, ensure_ascii=False))


def make_plots(pipeline, log, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t = log["t"]
    dt = config.SIM_TIMESTEP
    hp_t, hp = log["hand_pos_t"], log["hand_pos"]
    hq_t, hq = log["hand_quat_t"], log["hand_quat"]

    el = np.linalg.norm(hp[:, :3] - hp_t[:, :3], axis=1) * 1e3
    er = np.linalg.norm(hp[:, 3:] - hp_t[:, 3:], axis=1) * 1e3
    rl = np.rad2deg([np.linalg.norm(quat_error(hq_t[i, :4], hq[i, :4])) for i in range(len(t))])
    rr = np.rad2deg([np.linalg.norm(quat_error(hq_t[i, 4:], hq[i, 4:])) for i in range(len(t))])

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, el, label="left")
    ax[0].plot(t, er, label="right")
    ax[0].set_ylabel("position error [mm]")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, rl, label="left")
    ax[1].plot(t, rr, label="right")
    ax[1].set_ylabel("orientation error [deg]")
    ax[1].set_xlabel("time [s]")
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "hand_tracking_error.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax[0].plot(t, log["margin"])
    ax[0].axhline(0.0, color="k", lw=0.8, ls="--")
    ax[0].set_ylabel("CoM support margin [m]")
    ax[0].grid(alpha=0.3)
    drift = np.linalg.norm(log["foot_pos"] - log["foot_pos"][0], axis=1) * 1e3
    ax[1].plot(t, drift)
    ax[1].set_ylabel("foot drift [mm]")
    ax[1].set_xlabel("time [s]")
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "balance.png", dpi=140)
    plt.close(fig)

    qvel = log["qvel"][:, pipeline.robot.body_dof_adr]
    qacc = np.diff(qvel, axis=0) / dt
    qjerk = np.diff(qvel, n=2, axis=0) / dt ** 2
    ee_acc = np.diff(hp, n=2, axis=0) / dt ** 2
    ee_jerk = np.diff(hp, n=3, axis=0) / dt ** 3

    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    ax[0].plot(t, np.linalg.norm(qvel, axis=1))
    ax[0].set_ylabel("|joint vel| [rad/s]")
    ax[1].plot(t[:-1], np.linalg.norm(qacc, axis=1))
    ax[1].set_ylabel("|joint acc| [rad/s^2]")
    ax[2].plot(t[:-2], np.linalg.norm(ee_acc, axis=1))
    ax[2].set_ylabel("|EE acc| [m/s^2]")
    ax[2].set_xlabel("time [s]")
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "smoothness.png", dpi=140)
    plt.close(fig)
    _ = qjerk, ee_jerk

