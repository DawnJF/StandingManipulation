"""离线运动学碰撞检查：只规划 + 正运动学，不跑物理。

与 `sim.plan()` 的区别：规划步长可调（默认 10 ms，而不是 500 Hz），
因为这里只关心“参考轨迹是否穿模”，不需要物理稳定性。
一次全场景扫描在秒级完成。

输出：每个场景中臂（hand/wrist/elbow/shoulder）与躯干/髋/膝之间的接触对、
最大穿透深度和轨迹时间区间；脚-地接触不计。

用法：
    python -m g1_wbc.kin_check                # 全部场景
    python -m g1_wbc.kin_check fast10 circle  # 指定场景
    python -m g1_wbc.kin_check --dt 0.02      # 自定义规划步长
"""

import argparse
import collections
import time

import mujoco
import numpy as np

from .scenarios import SCENARIOS
from .sim import DemoPipeline
from .wbc import WholeBodyQP

FOOT_KEYS = ("ankle_roll", "ankle_pitch", "foot")
ARM_KEYS = ("hand_", "wrist_", "elbow", "shoulder_")
BODY_KEYS = ("hip_", "knee_", "pelvis", "waist_", "torso")


def _kind(b1, b2):
    def cls(name):
        if any(k in name for k in ARM_KEYS):
            return "arm"
        if any(k in name for k in BODY_KEYS):
            return "body"
        return "other"

    k1, k2 = cls(b1), cls(b2)
    if k1 == k2 == "arm":
        return "arm-arm"
    if "arm" in (k1, k2) and "body" in (k1, k2):
        return "arm-body"
    return None


def plan(pipe, dt):
    """按给定步长生成参考 qpos 序列（不开物理）。"""
    r, m = pipe.robot, pipe.robot.model
    dp = mujoco.MjData(m)
    dp.qpos[:] = r.data.qpos
    dp.qvel[:] = 0.0
    mujoco.mj_forward(m, dp)

    qp = WholeBodyQP(r, dp)
    qp.dt = dt
    posture = dp.qpos[r.body_qpos_adr].copy()
    n = int(round(pipe.total / dt))
    qpos = np.zeros((n, 7 + len(r.body_qpos_adr)))
    for k in range(n):
        t = k * dt
        tt = float(np.clip(t - pipe.pre, 0.0, pipe.traj.duration))
        hands = pipe.traj.hand(tt)
        dp.qvel[:] = 0.0
        mujoco.mj_forward(m, dp)
        dq = qp.solve(hands, pipe.traj.base_z(tt), pipe.com_target,
                      pipe.support_poly, posture,
                      support_lookahead=pipe.spec.support_lookahead,
                      arm_vel_limit=pipe.spec.arm_vel_limit,
                      base_vel_limit=pipe.spec.base_vel_limit,
                      base_angvel_limit=pipe.spec.base_angvel_limit,
                      torso_quat_target=pipe.traj.torso_quat(tt))
        full = np.zeros(m.nv)
        full[qp.cols] = dq
        mujoco.mj_integratePos(m, dp.qpos, full, 1.0)
        qpos[k, :7] = dp.qpos[:7]
        qpos[k, 7:] = dp.qpos[r.body_qpos_adr]
    return qpos


def stats_from_qpos(pipe, qpos, dt, stride=1):
    """对给定 qpos 序列统计非脚-地接触（不跑物理）。

    返回 {hit_frames, max_depth_mm, pairs: {pair: (kind, n, depth, t0, t1)}}。
    """
    r, m = pipe.robot, pipe.robot.model
    dp = mujoco.MjData(m)
    pairs = collections.Counter()
    depth = {}
    spans = {}
    hit_frames = 0
    for k in range(0, len(qpos), stride):
        dp.qpos[:7] = qpos[k, :7]
        dp.qpos[r.body_qpos_adr] = qpos[k, 7:]
        dp.qvel[:] = 0.0
        mujoco.mj_forward(m, dp)
        traj_t = k * dt - pipe.pre
        hit = False
        for c in range(dp.ncon):
            b1 = m.body(m.geom_bodyid[dp.contact[c].geom1]).name
            b2 = m.body(m.geom_bodyid[dp.contact[c].geom2]).name
            if any(f in b1 + b2 for f in FOOT_KEYS):
                continue
            kind = _kind(b1, b2)
            if kind is None:
                continue
            hit = True
            pair = tuple(sorted([b1, b2]))
            pairs[(kind, pair)] += 1
            depth[pair] = min(depth.get(pair, 0.0),
                              float(dp.contact[c].dist))
            spans.setdefault(pair, []).append(traj_t)
        if hit:
            hit_frames += 1
    detail = {}
    for (kind, pair), n in pairs.items():
        s = spans[pair]
        detail[pair] = (kind, n, depth[pair], min(s), max(s))
    return {
        "hit_frames": hit_frames,
        "frames": int(np.ceil(len(qpos) / stride)),
        "max_depth_mm": (min(depth.values()) * 1e3) if depth else 0.0,
        "pairs": detail,
    }


def scan(key, dt=0.01, show_clean=True):
    pipe = DemoPipeline(key)

    t0 = time.perf_counter()
    qpos = plan(pipe, dt)
    t_plan = time.perf_counter() - t0

    t0 = time.perf_counter()
    st = stats_from_qpos(pipe, qpos, dt)
    t_scan = time.perf_counter() - t0

    span = [x for (_, _, _, a, b) in st["pairs"].values() for x in (a, b)]
    span_txt = f"{min(span):+.2f}..{max(span):+.2f}s" if span else "-"
    print(f"| {key:10s} | {st['frames']:4d} | {st['hit_frames']:4d} "
          f"| {st['max_depth_mm']:+7.2f} | {len(st['pairs']):2d} | {span_txt} "
          f"| {t_plan:4.1f}s |", flush=True)
    if st["pairs"] and show_clean:
        items = sorted(st["pairs"].items(), key=lambda kv: -kv[1][1])
        for pair, (kind, n, d, a, b) in items:
            print(f"      [{kind}] {pair[0]} <-> {pair[1]}  n={n} "
                  f"({n*dt:.2f}s)  depth={d*1e3:+.2f}mm  "
                  f"traj_t={a:+.2f}..{b:+.2f}s", flush=True)
    return len(st["pairs"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenarios", nargs="*", default=None,
                    choices=None)
    ap.add_argument("--dt", type=float, default=0.01,
                    help="离线规划步长 [s]，默认 0.01")
    args = ap.parse_args()
    keys = args.scenarios or list(SCENARIOS)
    bad = [k for k in keys if k not in SCENARIOS]
    if bad:
        ap.error(f"未知场景 {', '.join(bad)}；可选：{', '.join(SCENARIOS)}")
    t0 = time.perf_counter()
    print(f"| 场景       | 帧数 | 穿透帧 | 最大穿透mm | 对数 | traj_t 区间 "
          f"| 规划 |")
    dirty = sum(scan(k, dt=args.dt) > 0 for k in keys)
    print(f"-- {len(keys)} 场景，{dirty} 个有接触，"
          f"总耗时 {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
