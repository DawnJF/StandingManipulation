"""FBX 骨骼动画 -> G1 双手 6-DOF 目标（take01~take03）。

数据源
------
data/fbx/g1_take.FBX / g1_take02.FBX / g1_take03.FBX：UE 风格人形骨架
（pelvis / spine / clavicle / upperarm / lowerarm / hand + 手指），24 fps。
本模块用纯 Python 解析二进制 FBX 7.x（zlib 数组），不新增第三方依赖。

坐标与比例
----------
源场景 GlobalSettings：UpAxis=+Z、FrontAxis=-Y、+X 为角色左侧（由脚趾朝向
与左右手位置实测确认）。G1 世界系为 x 向前、y 向左、z 向上，因此两张坐标系
之间是绕 z 轴 +90° 的刚体旋转 A：

    x_g1 = -y_fbx,  y_g1 = +x_fbx,  z_g1 = +z_fbx

（FBX 默认长度单位是厘米；GlobalSettings.UnitScaleFactor=1.0。）
比例按骨盆高度比 s = z_pelvis_robot / z_pelvis_source（≈0.96）统一缩放，
同时作用在位置与姿态（姿态只受坐标轴旋转影响，平移量乘 s）。

重定向方式
----------
位置：增量（offset）重定向，锚点为机器人静止站立时的手腕 site。源动画第 0
帧本来就是双臂自然下垂（三份 take 均如此），因此目标从机器人自己的自然站姿
出发，只复现源动作相对首帧的位移。这样避免了把源骨架的绝对姿态（例如手臂
更贴近躯干）硬套到 G1 上——G1 三指手在自然下垂时拇指距髋部只有几十毫米，
直接套用会稳定穿模。三份 take 的手臂前伸峰值处 CoM 裕度仍接近静止站立
（实测闭环 ≥ +17 mm），不需要再压缩幅度。

姿态：先取源手腕世界旋转的相对量 R_rel(t) = R(t) @ R(0)^T，再用 A 变换到
G1 世界系（R_g1 = A @ R_rel @ A^T），最后转成绕世界 x/y/z 的 RPY。整个
过程只用四元数/旋转矩阵：先做四元数符号连续化，再取连续旋转向量并等比
缩放到 ≤120°（源手腕峰值 180° 的人类旋前量，G1 腕部 roll 只有 ±113°，
硬追到 180° 会让 QP 用躯干/腿去凑姿态，破坏平衡），最后逐帧在两组等价欧拉解里选最连续的
一支，避免过万向节锁时出现 60°+ 的假跳变。

可达性：以静止肩点为球心做径向裁剪（半径 CLAMP_RADIUS），被裁剪的帧数与
最大越界量会记录下来并写进场景描述。三份 take 的裁剪量都很小（见 npz 里
的 clamp_frames / clamp_max_m），不需要裁掉片段。

缓存
----
提取结果写入 data/retarget/<key>.npz（每个约 6 KB）。运行时
`take_scenario(key)` 只读缓存，不解析 FBX；FBX 缺失时场景照常可用。
重建缓存：

    python -m g1_wbc.fbx_motion --build take01 take02 take03
"""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from . import config

SOURCE_DIR = config.PKG_ROOT / "data" / "fbx"
CACHE_DIR = config.PKG_ROOT / "data" / "retarget"

TAKE_FILES = {
    "take01": "g1_take.FBX",
    "take02": "g1_take02.FBX",
    "take03": "g1_take03.FBX",
}

KTIME_PER_SEC = 46186158000          # FBX KTime 单位
FRAME_STRIDE = 2                     # 每 2 帧取 1 个关键帧（24 fps -> 12 Hz）
CLAMP_RADIUS = 0.44                  # 手腕 site 到肩点的最大距离 [m]
MAX_WRIST_ROT_DEG = 120.0            # 腕部相对旋转上限（再大 QP 会借躯干凑姿态）
TAKE_LEG_STIFFNESS = 3.0             # 与 take 场景的 leg_stiffness_scale 一致
# 数据完整性原则：控制器评估必须使用原始 FBX 速度/时序。
# 禁止在这里按峰值手速做时间缩放或速度封顶来降低任务难度；
# 诊断用的慢速版本必须单独命名并明确标注，不能混入正式评估。
TAKE_TIME_SCALE = 1.0                # 原始速度，不做时间缩放


# --------------------------------------------------------------------------- #
# 最小 FBX 二进制解析
# --------------------------------------------------------------------------- #
class _Node:
    __slots__ = ("name", "props", "children")

    def __init__(self, name, props, children):
        self.name = name
        self.props = props
        self.children = children

    def first(self, name):
        for c in self.children:
            if c.name == name:
                return c
        return None

    def all(self, name):
        return [c for c in self.children if c.name == name]


def parse_fbx(path):
    """解析二进制 FBX（7.0+），返回顶层节点列表。"""
    buf = Path(path).read_bytes()
    if not buf.startswith(b"Kaydara FBX Binary"):
        raise ValueError(f"{path} 不是二进制 FBX（ASCII 版请先用 Blender 转存）")
    version = struct.unpack_from("<I", buf, 23)[0]
    big = version >= 7500
    sentinel = 25 if big else 13

    def read_prop(p):
        t = chr(buf[p])
        p += 1
        if t == "Y":
            return struct.unpack_from("<h", buf, p)[0], p + 2
        if t == "C":
            return bool(buf[p]), p + 1
        if t == "I":
            return struct.unpack_from("<i", buf, p)[0], p + 4
        if t == "F":
            return struct.unpack_from("<f", buf, p)[0], p + 4
        if t == "D":
            return struct.unpack_from("<d", buf, p)[0], p + 8
        if t == "L":
            return struct.unpack_from("<q", buf, p)[0], p + 8
        if t in "fdlib":
            n, enc, clen = struct.unpack_from("<III", buf, p)
            p += 12
            fmt = {"f": "f", "d": "d", "l": "q", "i": "i", "b": "b"}[t]
            size = struct.calcsize("<" + fmt)
            if enc:
                raw = zlib.decompress(buf[p:p + clen])
                p += clen
            else:
                raw = buf[p:p + n * size]
                p += n * size
            return list(struct.unpack_from("<" + fmt * n, raw, 0)), p
        if t == "S":
            n = struct.unpack_from("<I", buf, p)[0]
            return buf[p + 4:p + 4 + n].decode("utf-8", "replace"), p + 4 + n
        if t == "R":
            n = struct.unpack_from("<I", buf, p)[0]
            return buf[p + 4:p + 4 + n], p + 4 + n
        raise ValueError(f"未知 FBX 属性类型 {t!r} @{p}")

    def read_node(p):
        if big:
            end, nprops, _ = struct.unpack_from("<QQQ", buf, p)
            hdr = 24
        else:
            end, nprops, _ = struct.unpack_from("<III", buf, p)
            hdr = 12
        name_len = buf[p + hdr]
        if end == 0 and nprops == 0 and name_len == 0:
            return None, p + sentinel
        name_start = p + hdr + 1
        name = buf[name_start:name_start + name_len].decode("utf-8", "replace")
        p2 = name_start + name_len
        props = []
        for _ in range(nprops):
            v, p2 = read_prop(p2)
            props.append(v)
        children = []
        while p2 < end:
            child, p2 = read_node(p2)
            if child is None:
                break
            children.append(child)
        return _Node(name, props, children), end

    nodes, pos = [], 27
    while pos < len(buf) - 128:
        node, pos = read_node(pos)
        if node is None:
            break
        nodes.append(node)
    return nodes


# --------------------------------------------------------------------------- #
# 旋转工具
# --------------------------------------------------------------------------- #
def _euler_xyz_to_mat(e_deg):
    """FBX 默认 Lcl Rotation：R = Rz(yaw) @ Ry(pitch) @ Rx(roll)，角度为度。"""
    rx, ry, rz = np.deg2rad(np.asarray(e_deg, float))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    return np.array([
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx],
        [sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx],
        [-sy, cy * sx, cy * cx],
    ])


class FbxTake:
    """一张 FBX take 的骨骼层级与逐帧动画。"""

    def __init__(self, path):
        nodes = {n.name: n for n in parse_fbx(path)}
        objs = nodes["Objects"]
        self.models, self.curves, self.curve_nodes = {}, {}, {}
        for o in objs.children:
            if o.name == "Model":
                self.models[o.props[0]] = o
            elif o.name == "AnimationCurve":
                self.curves[o.props[0]] = o
            elif o.name == "AnimationCurveNode":
                self.curve_nodes[o.props[0]] = o

        self.name_of = {k: v.props[1].split("\x00")[0] for k, v in self.models.items()}
        self.by_name = {v: k for k, v in self.name_of.items()}

        # 骨骼父子关系：只保留 Model 之间（Deformer/NodeAttribute 连接会混进来）
        self.parent = {}
        op = []
        for c in nodes["Connections"].children:
            if c.name != "C":
                continue
            if c.props[0] == "OO":
                a, b = c.props[1], c.props[2]
                if a in self.models and b in self.models:
                    self.parent[a] = b
            else:
                op.append((c.props[1], c.props[2], c.props[3]))

        # 曲线节点 -> {轴: 曲线}；模型 -> {属性: 曲线节点}
        self.cn_axis, self.model_anim = {}, {}
        for a, b, p in op:
            if a in self.curves and b in self.curve_nodes and p.startswith("d|"):
                self.cn_axis.setdefault(b, {})[p[2:]] = self.curves[a]
            elif a in self.curve_nodes and b in self.models:
                self.model_anim.setdefault(b, {})[p] = a

        times = None
        for axes in self.cn_axis.values():
            if "X" in axes:
                key = axes["X"].first("KeyTime")
                times = np.asarray(key.props[0], float) / KTIME_PER_SEC
                break
        if times is None or len(times) < 2:
            raise ValueError(f"{path}: 没有找到动画曲线")
        self.times = times
        self.n_frames = len(times)

    # -- 属性读取 ----------------------------------------------------------- #
    def _props70(self, mid):
        out = {}
        p70 = self.models[mid].first("Properties70")
        if p70 is not None:
            for p in p70.children:
                if p.name == "P":
                    out[p.props[0]] = p.props[4:]
        return out

    def _channel(self, mid, prop, frame, default):
        cn = self.model_anim.get(mid, {}).get(prop)
        if cn is None:
            return np.asarray(default, float)
        vals = []
        for axis, d in zip("XYZ", default):
            curve = self.cn_axis.get(cn, {}).get(axis)
            vals.append(curve.first("KeyValueFloat").props[0][frame]
                        if curve is not None else d)
        return np.asarray(vals, float)

    def _local_matrix(self, mid, frame):
        """标准 FBX 局部变换：

        T * Roff * Rp * Rpre * R * Rpost^-1 * Rp^-1 * Soff * Sp * S * Sp^-1
        """
        p = self._props70(mid)
        t = self._channel(mid, "Lcl Translation", frame, p.get("Lcl Translation", [0, 0, 0])[:3])
        r = self._channel(mid, "Lcl Rotation", frame, p.get("Lcl Rotation", [0, 0, 0])[:3])
        s = self._channel(mid, "Lcl Scaling", frame, p.get("Lcl Scaling", [1, 1, 1])[:3])
        R = _euler_xyz_to_mat(r)
        pre = _euler_xyz_to_mat(p.get("PreRotation", [0, 0, 0])[:3])
        post = _euler_xyz_to_mat(p.get("PostRotation", [0, 0, 0])[:3])
        roff = np.asarray(p.get("RotationOffset", [0, 0, 0])[:3], float)
        rpiv = np.asarray(p.get("RotationPivot", [0, 0, 0])[:3], float)
        spiv = np.asarray(p.get("ScalingPivot", [0, 0, 0])[:3], float)
        soff = np.asarray(p.get("ScalingOffset", [0, 0, 0])[:3], float)

        def trans(v):
            M = np.eye(4)
            M[:3, 3] = v
            return M

        Rpre = np.eye(4)
        Rpre[:3, :3] = pre
        Rpost_inv = np.eye(4)
        Rpost_inv[:3, :3] = post.T
        Rl = np.eye(4)
        Rl[:3, :3] = R
        S = np.eye(4)
        S[:3, :3] = np.diag(s)

        return (trans(t) @ trans(roff) @ trans(rpiv) @ Rpre @ Rl @ Rpost_inv
                @ trans(-rpiv) @ trans(soff) @ trans(spiv) @ S @ trans(-spiv))

    def world_matrix(self, mid, frame):
        chain = []
        cur = mid
        while cur is not None and cur in self.models:
            chain.append(cur)
            cur = self.parent.get(cur)
        M = np.eye(4)
        for cur in reversed(chain):
            M = M @ self._local_matrix(cur, frame)
        return M

    def bone_name(self, pattern):
        for name, mid in self.by_name.items():
            if name == pattern:
                return mid
        raise KeyError(f"{pattern}: 骨架中没有该骨骼")

    def hand_series(self, side):
        """返回该侧手腕（hand 骨骼原点）的世界位置 [m] 与旋转矩阵，逐帧。"""
        mid = self.bone_name(f"hand_{'l' if side == 'left' else 'r'}")
        pos, rot = [], []
        for f in range(self.n_frames):
            M = self.world_matrix(mid, f)
            pos.append(M[:3, 3] * 0.01)                 # cm -> m
            rot.append(M[:3, :3].copy())
        return np.asarray(pos), np.asarray(rot)

    def bone_positions(self, name):
        mid = self.bone_name(name)
        return np.asarray([self.world_matrix(mid, f)[:3, 3] * 0.01
                           for f in range(self.n_frames)])


# --------------------------------------------------------------------------- #
# 重定向
# --------------------------------------------------------------------------- #
AXIS_MAP = np.array([[0.0, -1.0, 0.0],       # x_g1 = -y_fbx
                     [1.0, 0.0, 0.0],        # y_g1 = +x_fbx
                     [0.0, 0.0, 1.0]])       # z_g1 = +z_fbx


def _unwrap_deg(series):
    """对 (N,3) 的 RPY 角度逐列 unwrap，消除 ±180° 跳变。"""
    out = np.unwrap(np.deg2rad(series), axis=0)
    return np.rad2deg(out)


def _euler_zyx_deg(R):
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2]))
    roll = math.atan2(R[2, 1], R[2, 2])
    return np.rad2deg([roll, pitch, yaw])


def _continuous_rpy(rot_mats):
    """把旋转矩阵序列转成连续 RPY。

    同一姿态有 (r, p, y) 与 (r±180, 180-p, y±180) 两组等价解。源动作中手腕
    扫过 pitch≈±90° 的万向节锁，逐列 unwrap 会在两解之间产生 60°+ 的假跳变；
    这里逐帧选与上一帧角距离最小的分支，再做 360° 对齐。
    """
    out, prev = [], None
    for R in rot_mats:
        r, p, y = _euler_zyx_deg(R)
        best = None
        for cand in ((r, p, y), (r + 180.0, 180.0 - p, y + 180.0)):
            if prev is None:
                aligned = np.asarray(cand)
            else:
                aligned = np.asarray(
                    [cand[i] + 360.0 * round((prev[i] - cand[i]) / 360.0)
                     for i in range(3)])
            cost = float(np.abs(aligned - prev).sum()) if prev is not None else 0.0
            if best is None or cost < best[0]:
                best = (cost, aligned)
        out.append(best[1])
        prev = best[1]
    return np.asarray(out)


@dataclass
class RetargetResult:
    key: str
    times: tuple            # 源时间轴（秒，未缩放）
    time_scale: float
    left_pos: tuple
    left_rpy: tuple
    right_pos: tuple
    right_rpy: tuple
    source_duration: float
    peak_speed: float       # 原始峰值手速 [m/s]
    clamp_frames: int
    clamp_max_m: float
    scale: float
    description: str

    def save(self, path):
        np.savez_compressed(
            path,
            times=np.asarray(self.times, float),
            left_pos=np.asarray(self.left_pos, float),
            left_rpy=np.asarray(self.left_rpy, float),
            right_pos=np.asarray(self.right_pos, float),
            right_rpy=np.asarray(self.right_rpy, float),
            time_scale=np.array(self.time_scale),
            source_duration=np.array(self.source_duration),
            peak_speed=np.array(self.peak_speed),
            clamp_frames=np.array(self.clamp_frames),
            clamp_max_m=np.array(self.clamp_max_m),
            scale=np.array(self.scale),
            key=np.array(self.key),
            description=np.array(self.description),
        )


def build_take(key, robot=None, verbose=True):
    """解析 FBX、重定向并写入缓存，返回 RetargetResult。

    锚点必须与场景运行时的站姿完全一致（同样的腿部刚度与稳定时间），
    否则目标会带上十几毫米的系统性偏差。
    """
    import mujoco
    from .model import RobotModel

    if key not in TAKE_FILES:
        raise KeyError(f"未知 take {key!r}，可选：{', '.join(TAKE_FILES)}")
    src = SOURCE_DIR / TAKE_FILES[key]
    if not src.exists():
        raise FileNotFoundError(f"缺少源文件 {src}")

    take = FbxTake(src)
    if robot is None:
        robot = RobotModel(leg_gain_scale=TAKE_LEG_STIFFNESS)
    m, d = robot.model, robot.data

    pelvis = take.bone_positions("pelvis")
    scale = float(d.qpos[2] / pelvis[:, 2].mean())

    rails, clamped = {}, {}
    for side in ("left", "right"):
        p_src, R_src = take.hand_series(side)
        rest_site = robot.hand_anchor[f"{side}_hand"][0]

        # 世界位置增量 -> G1 世界，锚定到机器人自己的自然手腕位
        pos = rest_site + (p_src - p_src[0]) @ AXIS_MAP.T * scale

        # 超可达部分：以静止肩点为球心径向裁剪（保留方向，只收半径）
        sh_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,
                                  f"{side}_shoulder_pitch_link")
        shoulder = d.xpos[sh_id].copy()
        vec = pos - shoulder
        dist = np.linalg.norm(vec, axis=1)
        over = dist > CLAMP_RADIUS
        clamped[side] = (int(over.sum()),
                         float((dist[over] - CLAMP_RADIUS).max()) if over.any() else 0.0)
        pos[over] = shoulder + vec[over] * (CLAMP_RADIUS / dist[over])[:, None]

        # 世界姿态增量 -> G1 世界。源手腕单段最大相对旋转可达 175°（人类
        # 前臂旋前 + 腕伸），G1 腕部（roll ±113°、pitch/yaw ±92°）做不到，
        # 也不该让 QP 去追一个 180° 的反向解；按最大值等比例缩放。
        # 关键：先在四元数空间做符号连续化，再转成连续旋转向量后缩放——
        # 直接对 as_rotvec() 缩放会在源旋转跨过 180° 时把轴翻到反向，
        # 造成 60°+ 的假跳变。
        Rrel = np.array([AXIS_MAP @ (R @ R_src[0].T) @ AXIS_MAP.T for R in R_src])
        angles = np.degrees(np.arccos(
            np.clip((np.trace(Rrel, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)))
        rot_scale = min(1.0, MAX_WRIST_ROT_DEG / max(float(angles.max()), 1e-6))
        quat = Rotation.from_matrix(Rrel).as_quat()          # xyzw
        for i in range(1, len(quat)):
            if np.dot(quat[i], quat[i - 1]) < 0:
                quat[i] = -quat[i]
        vec = quat[:, :3]
        vn = np.linalg.norm(vec, axis=1)
        theta = 2.0 * np.arctan2(vn, quat[:, 3])
        axis = np.divide(vec, vn[:, None], out=np.zeros_like(vec), where=vn[:, None] > 1e-9)
        rotvec = axis * theta[:, None] * rot_scale
        rails[side] = (pos, _continuous_rpy(Rotation.from_rotvec(rotvec).as_matrix()),
                       float(angles.max()), rot_scale)

    # 关键帧抽样（含末帧），12 Hz
    idx = list(range(0, take.n_frames, FRAME_STRIDE))
    if idx[-1] != take.n_frames - 1:
        idx.append(take.n_frames - 1)
    times = take.times[idx]

    peak = 0.0
    for side in ("left", "right"):
        pos = rails[side][0][idx]
        peak = max(peak, float(np.max(
            np.linalg.norm(np.diff(pos, axis=0), axis=1) / np.diff(times))))
    time_scale = float(TAKE_TIME_SCALE)

    clamp_frames = sum(v[0] for v in clamped.values())
    clamp_max = max(v[1] for v in clamped.values())
    rot_max = max(rails["left"][2], rails["right"][2])
    rot_scale = min(rails["left"][3], rails["right"][3])
    res = RetargetResult(
        key=key,
        times=tuple(float(t) for t in times),
        time_scale=time_scale,
        left_pos=tuple(tuple(float(v) for v in p) for p in rails["left"][0][idx]),
        left_rpy=tuple(tuple(float(v) for v in r) for r in rails["left"][1][idx]),
        right_pos=tuple(tuple(float(v) for v in p) for p in rails["right"][0][idx]),
        right_rpy=tuple(tuple(float(v) for v in r) for r in rails["right"][1][idx]),
        source_duration=float(take.times[-1]),
        peak_speed=peak,
        clamp_frames=clamp_frames,
        clamp_max_m=clamp_max,
        scale=scale,
        description="",
    )
    rot_text = (f"腕部相对旋转峰值 {rot_max:.0f}° 按 {rot_scale:.2f}× 缩放到 "
                f"{rot_max * rot_scale:.0f}°（源手腕旋前量超出 G1 腕部 roll 范围）"
                if rot_scale < 0.999 else
                f"腕部相对旋转峰值 {rot_max:.0f}° 完整保留")
    res.description = (
        f"FBX 动作重定向（{TAKE_FILES[key]}）：源时长 {res.source_duration:.2f} s、"
        f"原始速度（未做时间缩放），峰值手速 {peak:.2f} m/s；"
        f"位置按首帧增量重定向（比例 {scale:.3f}），姿态取四元数相对量并转 RPY；"
        f"{rot_text}；"
        f"超可达部分以肩点为球心径向裁剪（{clamp_frames} 帧，"
        f"最大越界 {clamp_max * 1e3:.0f} mm）。")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    res.save(CACHE_DIR / f"{key}.npz")
    if verbose:
        print(f"[retarget] {key}: {len(res.times)} 关键帧, 时长 "
              f"{res.source_duration:.2f} s（原始速度）, 比例 {scale:.3f}, "
              f"峰值手速 {peak:.2f} m/s, "
              f"腕部旋转 {rot_max:.0f}° x{rot_scale:.2f}, "
              f"裁剪 {clamp_frames} 帧 (max {clamp_max * 1e3:.0f} mm)")
    return res


TAKE_TITLES = {
    "take01": "FBX take01 · 右手抬起挥动",
    "take02": "FBX take02 · 双手前推",
    "take03": "FBX take03 · 右手前上举",
}


def take_scenario(key):
    """从缓存构造 Scenario；缓存缺失时自动从 FBX 重建。"""
    if key not in TAKE_FILES:
        raise KeyError(f"未知 take {key!r}，可选：{', '.join(TAKE_FILES)}")
    path = CACHE_DIR / f"{key}.npz"
    if not path.exists():
        build_take(key, verbose=True)
    data = np.load(path, allow_pickle=False)

    from .scenarios import Scenario  # 延迟导入，避免循环依赖

    return Scenario(
        key=key,
        title=TAKE_TITLES[key],
        description=str(data["description"]),
        times=tuple(float(t) for t in data["times"]),
        time_scale=float(data["time_scale"]),
        pre_hold=0.8,
        post_hold=1.0,
        interp="cubic",
        leg_stiffness_scale=3.0,
        arm_vel_limit=8.0,
        left_pos=tuple(tuple(float(v) for v in row) for row in data["left_pos"]),
        left_rpy_deg=tuple(tuple(float(v) for v in row) for row in data["left_rpy"]),
        right_pos=tuple(tuple(float(v) for v in row) for row in data["right_pos"]),
        right_rpy_deg=tuple(tuple(float(v) for v in row) for row in data["right_rpy"]),
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="FBX 双手轨迹重定向")
    ap.add_argument("--build", nargs="+", choices=list(TAKE_FILES), default=[],
                    help="解析 FBX 并重建缓存")
    ap.add_argument("--probe", nargs="+", choices=list(TAKE_FILES), default=[],
                    help="只打印 take 的骨架与轨迹统计")
    args = ap.parse_args()

    if args.probe:
        for key in args.probe:
            take = FbxTake(SOURCE_DIR / TAKE_FILES[key])
            print(f"== {key} ({TAKE_FILES[key]}): {take.n_frames} 帧, "
                  f"{take.times[-1]:.3f} s")
            for side in ("left", "right"):
                p, _ = take.hand_series(side)
                v = np.linalg.norm(np.diff(p, axis=0), axis=1) / np.diff(take.times)
                print(f"   {side:5s} start={np.round(p[0], 3)} end={np.round(p[-1], 3)} "
                      f"z {p[:, 2].min():.2f}..{p[:, 2].max():.2f} "
                      f"peak {v.max():.2f} m/s")
    if args.build:
        for key in args.build:
            build_take(key)
