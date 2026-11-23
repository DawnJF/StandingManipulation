"""四元数与二维凸多边形工具。"""

import numpy as np
import mujoco


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q)


def quat_mul(a, b):
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, np.asarray(a, float), np.asarray(b, float))
    return quat_normalize(out)


def quat_conj(q):
    out = np.zeros(4)
    mujoco.mju_negQuat(out, np.asarray(q, float))
    return out


def quat_error(target, current):
    """返回把 current 旋转到 target 的世界系旋转向量（轴 * 角）。"""
    qe = quat_mul(target, quat_conj(current))
    if qe[0] < 0.0:
        qe = -qe
    vel = np.zeros(3)
    mujoco.mju_quat2Vel(vel, qe, 1.0)
    return vel


def quat_to_mat(q):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, np.asarray(q, float))
    return mat.reshape(3, 3)


def mat_to_quat(mat):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(mat, float).reshape(9))
    return quat_normalize(q)


def quat_slerp(q0, q1, u):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    if dot > 0.9995:
        return quat_normalize((1.0 - u) * q0 + u * q1)
    theta = np.arccos(np.clip(dot, -1.0, 1.0))
    sin_theta = np.sin(theta)
    return quat_normalize(np.sin((1.0 - u) * theta) / sin_theta * q0
                          + np.sin(u * theta) / sin_theta * q1)


def rotvec_to_quat(v):
    v = np.asarray(v, float)
    angle = np.linalg.norm(v)
    if angle < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q = np.zeros(4)
    mujoco.mju_axisAngle2Quat(q, v / angle, angle)
    return quat_normalize(q)


def convex_hull_2d(points):
    """单调链凸包，返回逆时针顶点。"""
    pts = sorted({(float(p[0]), float(p[1])) for p in points})
    if len(pts) <= 2:
        return np.asarray(pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.asarray(lower[:-1] + upper[:-1])


def torso_pitch_deg(robot, data):
    """躯干世界系前倾角 [deg]（正值为前倾）。"""
    R = data.xmat[robot.body_ids["torso"]].reshape(3, 3)
    return float(-np.degrees(np.arctan2(R[2, 0], R[0, 0])))


def polygon_centroid(poly):
    """多边形面积质心（逆时针/顺时针均可）。

    不能用顶点平均代替：凸包顶点在某一侧更密时（脚掌前端常有 4 个
    接触球、后端只有 2 个），顶点平均会明显偏向该侧，把 CoM 目标带偏。
    """
    p = np.asarray(poly, float)
    if len(p) < 3:
        return p.mean(axis=0)
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cross = x * y1 - x1 * y
    area = 0.5 * cross.sum()
    if abs(area) < 1e-12:
        return p.mean(axis=0)
    return np.array([((x + x1) * cross).sum() / (6.0 * area),
                     ((y + y1) * cross).sum() / (6.0 * area)])


def polygon_edges(poly):
    """返回每条边的单位内法向与偏移（多边形逆时针）。"""
    normals, offsets = [], []
    n = len(poly)
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        edge = b - a
        normal = np.array([edge[1], -edge[0]])
        length = np.linalg.norm(normal)
        if length < 1e-12:
            continue
        normal /= length
        normals.append(normal)
        offsets.append(float(np.dot(normal, a)))
    return np.asarray(normals), np.asarray(offsets)


def polygon_shrink(poly, margin):
    normals, offsets = polygon_edges(poly)
    # 对每条边向内平移 margin，重新求相邻边交点
    shifted = offsets - margin
    n = len(normals)
    result = []
    for i in range(n):
        j = (i - 1) % n
        a = np.array([normals[j][0], normals[j][1]])
        b = np.array([normals[i][0], normals[i][1]])
        rhs = np.array([shifted[j], shifted[i]])
        mat = np.stack([a, b])
        if abs(np.linalg.det(mat)) < 1e-9:
            continue
        result.append(np.linalg.solve(mat, rhs))
    return np.asarray(result)


def polygon_signed_margin(poly, point):
    """点到凸多边形的有符号距离，内部为正。"""
    normals, offsets = polygon_edges(poly)
    return float(np.min(offsets - normals @ np.asarray(point, float)))


def smoothstep_minjerk(u):
    u = np.clip(u, 0.0, 1.0)
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5
