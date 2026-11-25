"""PICO 手柄实时遥操作 G1：TCP 接收 + MuJoCo 实时跟踪。

数据链路：PICO OpenXR app --(TCP, adb reverse)--> 本机 2233 端口。
报文与 pico_tele_openxr 的 send_action 一致：
    ms:lx,ly,lz,lqx,lqy,lqz,lqw,ltrigger,
       rx,ry,rz,rqx,rqy,rqz,rqw,rtrigger,
       left_x,left_y,right_a,right_b

用法：
    adb reverse tcp:2233 tcp:2233
    PYTHONPATH=src mjpython -m g1_wbc.pico_live --monitor   # 只看手柄位置
    PYTHONPATH=src mjpython -m g1_wbc.pico_live             # G1 双手跟随（WBC）

按 A 键接管/释放。接管后手柄相对位置映射到机器人双手目标上。
"""

import argparse
import os
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

from . import realtime
from .sim import DemoPipeline
from .utils import mat_to_quat, quat_to_mat

HANDS = ("left_hand", "right_hand")
SIDES = {"left_hand": "left", "right_hand": "right"}


# ---------------------------------------------------------------------- #
# 坐标系：OpenXR LOCAL (y 上, -z 前, 米) -> MuJoCo (z 上, x 前)
# ---------------------------------------------------------------------- #
def _basis(yaw_deg=0.0):
    """返回 3x3 映射矩阵：p_robot = M @ p_pico。"""
    m = np.array(
        [
            [0.0, 0.0, -1.0],  # robot x = -pico z（向前）
            [-1.0, 0.0, 0.0],  # robot y = -pico x（向左）
            [0.0, 1.0, 0.0],  # robot z =  pico y（向上）
        ]
    )
    if abs(yaw_deg) > 1e-9:
        c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
        rot_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        m = rot_z @ m
    return m


def _rot_to_robot(rot_mat, basis):
    return basis @ rot_mat @ basis.T


# ---------------------------------------------------------------------- #
# TCP 接收端
# ---------------------------------------------------------------------- #
class PicoReceiver:
    """监听 2233 端口，缓存最新一帧手柄位姿。"""

    def __init__(self, host="0.0.0.0", port=2233):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((host, port))
        self._sock.listen(4)
        self._sock.settimeout(0.2)
        self._lock = threading.Lock()
        self._snap = None
        self._count = 0
        self._last_recv = 0.0
        self._client_addr = None
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self):
        self._thread.start()

    def close(self):
        self._running = False
        try:
            self._sock.close()
        except OSError:
            pass

    # ------------------------------------------------------------------ #
    def _loop(self):
        client = None
        buf = b""
        while self._running:
            if client is None:
                try:
                    client, addr = self._sock.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break
                client.settimeout(0.5)
                buf = b""
                self._client_addr = f"{addr[0]}:{addr[1]}"
                print(f"[PICO] 客户端已连接 {self._client_addr}", flush=True)
            try:
                chunk = client.recv(8192)
                if not chunk:
                    raise ConnectionError("closed")
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if line.strip():
                        self._handle(line)
            except socket.timeout:
                continue
            except OSError:
                print("[PICO] 客户端断开，等待重连 ...", flush=True)
                try:
                    client.close()
                except OSError:
                    pass
                client = None
                self._client_addr = None

    def _handle(self, line: bytes):
        text = line.decode("utf-8", "ignore").strip().rstrip(",")
        if ":" not in text:
            return
        stamp_text, payload = text.split(":", 1)
        parts = payload.split(",")
        if len(parts) != 20:
            return
        try:
            v = [float(x) for x in parts]
            stamp = int(stamp_text)
        except ValueError:
            return

        snap = {
            "stamp": stamp,
            "left": (np.array(v[0:3]), np.array(v[3:7])),  # pos, quat(xyzw)
            "right": (np.array(v[8:11]), np.array(v[11:15])),
            "left_trigger": v[7],
            "right_trigger": v[15],
            "buttons": {"x": v[16], "y": v[17], "a": v[18], "b": v[19]},
            "recv_t": time.perf_counter(),
        }
        with self._lock:
            self._snap = snap
            self._count += 1
            self._last_recv = snap["recv_t"]

    def latest(self):
        """返回最新快照；超过 0.5 s 没数据则返回 None。"""
        with self._lock:
            snap = self._snap
        if snap is None or time.perf_counter() - snap["recv_t"] > 0.5:
            return None
        return snap

    @property
    def count(self):
        with self._lock:
            return self._count

    @property
    def client_addr(self):
        return self._client_addr


# ---------------------------------------------------------------------- #
# 遥操作映射
# ---------------------------------------------------------------------- #
class PicoTeleop:
    def __init__(self, pipe, basis, scale=1.0, use_rotation=True):
        self.pipe = pipe
        self.basis = basis
        self.scale = scale
        self.use_rotation = use_rotation
        self.engaged = False
        self._prev_a = False
        self._anchor = None
        self._last_state_print = 0.0

    def _robot_hand(self, key):
        r, d = self.pipe.robot, self.pipe.robot.data
        return d.site_xpos[r.site_ids[key]].copy(), r.site_quat(key)

    def hold_targets(self):
        return {key: self._robot_hand(key) for key in HANDS}

    def _make_anchor(self, snap):
        anchor = {}
        for key in HANDS:
            side = SIDES[key]
            pico_pos, pico_quat = snap[side]
            robot_pos, robot_quat = self._robot_hand(key)
            anchor[key] = {
                "pico_pos": pico_pos.copy(),
                "pico_quat": pico_quat.copy(),
                "robot_pos": robot_pos,
                "robot_quat": robot_quat,
            }
        self._anchor = anchor

    def _map(self, key, snap):
        side = SIDES[key]
        pico_pos, pico_quat = snap[side]
        a = self._anchor[key]

        delta_pos = self.basis @ (pico_pos - a["pico_pos"]) * self.scale
        target_pos = a["robot_pos"] + delta_pos

        if self.use_rotation:
            r_pico = quat_to_mat(
                np.array([pico_quat[3], pico_quat[0], pico_quat[1], pico_quat[2]])
            )
            r_ref = quat_to_mat(
                np.array(
                    [
                        a["pico_quat"][3],
                        a["pico_quat"][0],
                        a["pico_quat"][1],
                        a["pico_quat"][2],
                    ]
                )
            )
            r_delta = _rot_to_robot(r_pico @ r_ref.T, self.basis)
            r_target = r_delta @ quat_to_mat(a["robot_quat"])
            target_quat = mat_to_quat(r_target)
        else:
            target_quat = a["robot_quat"]
        return target_pos, target_quat

    def target(self, snap, t=None, _step=None):
        if snap is None:
            return self.hold_targets()

        a_pressed = bool(snap["buttons"]["a"])
        if a_pressed and not self._prev_a:
            if self.engaged:
                self.engaged = False
                self._anchor = None
                print("\n[PICO] 已释放，机器人保持当前目标", flush=True)
            elif self._last_packet_fresh(snap):
                self._make_anchor(snap)
                self.engaged = True
                print("\n[PICO] 已接管：双手跟随手柄（按 A 释放）", flush=True)
        self._prev_a = a_pressed

        if not self.engaged:
            return self.hold_targets()
        if not self._last_packet_fresh(snap):
            print("[PICO] 数据超时，暂停更新", flush=True)
            return self.hold_targets()
        return {key: self._map(key, snap) for key in HANDS}

    @staticmethod
    def _last_packet_fresh(snap):
        return time.perf_counter() - snap["recv_t"] <= 0.5


# ---------------------------------------------------------------------- #
# 启动辅助
# ---------------------------------------------------------------------- #
def _ensure_viewer_runtime(module="g1_wbc.pico_live"):
    """macOS 下用 mjpython 重启自己，否则 MuJoCo 窗口无法交互。"""
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
    os.execvpe(str(mjpython), [str(mjpython), "-m", module, *sys.argv[1:]], env)
    return True


def _print_banner(controller, monitor):
    print("\n=== PICO 遥操作 ===", flush=True)
    print("  1) adb reverse tcp:2233 tcp:2233（PC 上执行一次）", flush=True)
    print("  2) 头盔打开 OpenXR_Samples，戴好手柄", flush=True)
    if monitor:
        print("  监控模式：只显示手柄映射后的目标位置，不动机器人。", flush=True)
    else:
        print(f"  G1 控制器：{controller}；按 A 接管/释放双手。", flush=True)
    print("  关闭窗口即退出。\n", flush=True)


# ---------------------------------------------------------------------- #
# 监控模式：只把映射后的手柄位置显示成 mocap 标记
# ---------------------------------------------------------------------- #
def run_monitor(args, pipe, receiver, basis):
    import mujoco
    import mujoco.viewer

    r, d = pipe.robot, pipe.robot.data
    rp0 = {key: d.site_xpos[r.site_ids[key]].copy() for key in HANDS}
    anchor = None
    last_print = 0.0

    realtime.init_markers(pipe)
    with mujoco.viewer.launch_passive(r.model, d) as viewer:
        pipe.set_camera(viewer.cam)
        while viewer.is_running():
            snap = receiver.latest()
            if snap is not None:
                if anchor is None:
                    anchor = {key: snap[SIDES[key]] for key in HANDS}
                    print("[PICO] 收到数据，开始显示手柄位姿", flush=True)
                for key in HANDS:
                    pico_pos, _ = snap[SIDES[key]]
                    a = anchor[key][0]
                    pos = rp0[key] + basis @ (pico_pos - a) * args.scale
                    pos = np.clip(pos, [-0.25, -0.60, 0.30], [0.85, 0.60, 1.55])
                    mid = pipe.mocap_ids[key]
                    d.mocap_pos[mid] = pos
                    d.mocap_quat[mid] = r.site_quat(key)
                mujoco.mj_forward(r.model, d)
            now = time.perf_counter()
            if now - last_print > 2.0:
                n = receiver.count
                if snap is None:
                    print(
                        f"[PICO] 等待数据 ... 累计 {n} 帧，客户端 {receiver.client_addr}",
                        flush=True,
                    )
                elif anchor is not None:
                    lp = snap["left"][0]
                    rp = snap["right"][0]
                    print(
                        f"[PICO] {n} 帧  L=({lp[0]:+.2f},{lp[1]:+.2f},{lp[2]:+.2f}) "
                        f"R=({rp[0]:+.2f},{rp[1]:+.2f},{rp[2]:+.2f})",
                        flush=True,
                    )
                last_print = now
            viewer.sync()
            time.sleep(1.0 / 60.0)


# ---------------------------------------------------------------------- #
# 主模式：G1 WBC 双手跟随
# ---------------------------------------------------------------------- #
def run_follow(args, pipe, receiver, basis):
    import mujoco
    import mujoco.viewer

    r, d = pipe.robot, pipe.robot.data
    q0 = d.qpos.copy()
    teleop = PicoTeleop(
        pipe, basis, scale=args.scale, use_rotation=not args.no_rotation
    )

    stat = {"t": time.perf_counter(), "steps": 0, "count": receiver.count}

    def target_fn(step, t):
        stat["steps"] += 1
        now = time.perf_counter()
        if now - stat["t"] >= 4.0:
            wall = now - stat["t"]
            n = receiver.count
            print(
                f"[PICO] 仿真 {stat['steps'] * pipe.dt / wall * 100:5.0f}% 实时 | "
                f"手柄 {(n - stat['count']) / wall:4.0f} Hz | "
                f"{'已接管' if teleop.engaged else '待机'}",
                flush=True,
            )
            stat.update(t=now, steps=0, count=n)
        return teleop.target(receiver.latest(), t, step)

    base_z_fn = realtime.standing_base_z(pipe)
    # 遥操作要跟手：目标滤波不能像 mocap 拖动那样保守
    shape = dict(
        target_omega=40.0,
        target_speed=2.0,
        margin_guard=0.012,
        lean_gain=-0.22,
        com_margin=0.015,
        max_target_acc=20.0,
        rot_tau=0.025,
        predictive_guard=True,
        com_vel_gain=2.5,
        com_vel_tau=0.10,
        target_bounds=((-0.20, 0.75), (-0.55, 0.55), (0.35, 1.50)),
        # 向上原来只给 +0.20 m（初始手高 0.684 -> 0.884，只到腰）；
        # 实测 +0.70 m（手到 1.384 m，接近头）双手同时抬仍稳定。
        bounds_rel=((-0.20, 0.25), (-0.25, 0.25), (-0.35, 0.70)),
        log_every=5,
    )
    control_every = args.control_every

    res = None
    with mujoco.viewer.launch_passive(r.model, d) as viewer:
        pipe.set_camera(viewer.cam)
        while viewer.is_running():
            d.qpos[:] = q0
            d.qvel[:] = 0.0
            d.ctrl[:] = 0.0
            mujoco.mj_forward(r.model, d)
            res = realtime.run(
                pipe,
                target_fn,
                base_z_fn,
                n_steps=10**9,
                viewer=viewer,
                pace=True,
                verbose=False,
                drive_markers=True,
                controller=args.controller,
                control_every=control_every,
                sync_dt=1.0 / max(args.viewer_hz, 1),
                **shape,
            )
            break
    if res:
        s = realtime.summarize(
            res, [r.foot_anchor["left_foot"], r.foot_anchor["right_foot"]]
        )
        print(
            f"[PICO] 结束：{s.get('duration_s', 0.0):.1f} s  "
            f"{'摔倒' if s.get('fell') else '站立'}",
            flush=True,
        )


def main():
    ap = argparse.ArgumentParser(description="PICO 手柄实时遥操作 G1")
    ap.add_argument("--scenario", default="demo")
    ap.add_argument(
        "--controller", choices=("wbc", "balance", "kinematic"), default="wbc"
    )
    ap.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="手柄位移到机器人位移的缩放（默认 1.0）",
    )
    ap.add_argument(
        "--yaw", type=float, default=0.0, help="手柄前向相对机器人前向的偏航角（度）"
    )
    ap.add_argument(
        "--monitor", action="store_true", help="只显示手柄映射位置，不动机器人"
    )
    ap.add_argument("--no-rotation", action="store_true", help="只跟随位置，不跟随姿态")
    ap.add_argument(
        "--control-every",
        type=int,
        default=5,
        help="WBC 每 N 个物理步解一次（默认 5，约 100 Hz；越小越跟手越吃 CPU）",
    )
    ap.add_argument(
        "--viewer-hz",
        type=float,
        default=60.0,
        help="MuJoCo 窗口刷新率（默认 60 Hz；机器较慢时降到 30）",
    )
    ap.add_argument("--port", type=int, default=2233)
    args = ap.parse_args()

    if not _ensure_viewer_runtime():
        print("未找到 mjpython（应与 python 在同一目录）。", file=sys.stderr)
        sys.exit(1)

    _print_banner(args.controller, args.monitor)
    basis = _basis(args.yaw)
    pipe = DemoPipeline(args.scenario)
    receiver = PicoReceiver(port=args.port)
    receiver.start()
    try:
        if args.monitor:
            run_monitor(args, pipe, receiver, basis)
        else:
            run_follow(args, pipe, receiver, basis)
    except KeyboardInterrupt:
        print("\n已退出。")
    finally:
        receiver.close()


if __name__ == "__main__":
    main()
