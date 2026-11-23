"""MuJoCo 模型封装：索引、站立初始化、足端支撑多边形。"""

from pathlib import Path

import mujoco
import numpy as np

from . import config
from .utils import convex_hull_2d, mat_to_quat, polygon_edges

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
WAIST_JOINTS = ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
LEFT_ARM_JOINTS = [
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]
RIGHT_ARM_JOINTS = [name.replace("left_", "right_") for name in LEFT_ARM_JOINTS]
BODY_JOINTS = LEG_JOINTS + WAIST_JOINTS + LEFT_ARM_JOINTS + RIGHT_ARM_JOINTS

LEFT_FINGER_JOINTS = [
    "left_hand_thumb_0_joint", "left_hand_thumb_1_joint", "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint", "left_hand_middle_1_joint",
    "left_hand_index_0_joint", "left_hand_index_1_joint",
]
RIGHT_FINGER_JOINTS = [name.replace("left_", "right_") for name in LEFT_FINGER_JOINTS]
FINGER_JOINTS = LEFT_FINGER_JOINTS + RIGHT_FINGER_JOINTS


class RobotModel:
    """G1 + 手部末端 site 的模型索引与运动学辅助。"""

    def __init__(self, xml_path=None, settle_time=2.0, leg_gain_scale=1.0):
        self.xml_path = Path(xml_path) if xml_path else config.SCENE_XML
        self.model = mujoco.MjModel.from_xml_path(str(self.xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = config.SIM_TIMESTEP

        self._build_indices()
        self._scale_leg_gains(leg_gain_scale)
        self.reset_to_stand(settle_time)

    def _scale_leg_gains(self, scale):
        """按场景需要提高腿部位置伺服刚度（kp 与 kv 同步放大）。"""
        if scale == 1.0:
            return
        m = self.model
        acts = np.array([self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEG_JOINTS])
        m.actuator_gainprm[acts, 0] *= scale
        m.actuator_biasprm[acts, 1] *= scale
        m.actuator_biasprm[acts, 2] *= np.sqrt(scale)

    def _id(self, obj_type, name):
        idx = mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise ValueError(f"找不到 {name}")
        return idx

    def _head_geom(self):
        """头部网格 geom：G1 没有独立颈关节，头是 torso 上的固定点。"""
        mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, "head_link")
        for gid in range(self.model.ngeom):
            if (self.model.geom_dataid[gid] == mesh_id
                    and self.model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_MESH):
                return int(gid)
        return -1

    def _build_indices(self):
        m = self.model
        self.site_ids = {
            "left_hand": self._id(mujoco.mjtObj.mjOBJ_SITE, "left_hand_ee"),
            "right_hand": self._id(mujoco.mjtObj.mjOBJ_SITE, "right_hand_ee"),
            "left_foot": self._id(mujoco.mjtObj.mjOBJ_SITE, "left_foot"),
            "right_foot": self._id(mujoco.mjtObj.mjOBJ_SITE, "right_foot"),
        }
        self.body_ids = {
            "torso": self._id(mujoco.mjtObj.mjOBJ_BODY, "torso_link"),
            "left_ankle": self._id(mujoco.mjtObj.mjOBJ_BODY, "left_ankle_roll_link"),
            "right_ankle": self._id(mujoco.mjtObj.mjOBJ_BODY, "right_ankle_roll_link"),
        }
        self.head_geom = self._head_geom()

        self.body_joint_ids = [self._id(mujoco.mjtObj.mjOBJ_JOINT, n) for n in BODY_JOINTS]
        self.body_qpos_adr = np.array([m.jnt_qposadr[j] for j in self.body_joint_ids])
        self.body_dof_adr = np.array([m.jnt_dofadr[j] for j in self.body_joint_ids])
        self.body_act_ids = np.array([
            self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in BODY_JOINTS
        ])
        self.finger_act_ids = np.array([
            self._id(mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in FINGER_JOINTS
        ])

        # 三指手默认握拳：拇指收拢贴向掌心。张开的拇指在髋部/大腿附近
        # 扫过时（shelf/pickup 等）会与身体网格相交；收拢后静止间隙由
        # 46 mm 提高到约 94 mm。中指/食指保持伸展。
        self.finger_ctrl = np.zeros(len(FINGER_JOINTS))
        for i, name in enumerate(FINGER_JOINTS):
            if "thumb_1" in name:
                self.finger_ctrl[i] = 0.50
            elif "thumb_2" in name:
                self.finger_ctrl[i] = 1.50
            if name.startswith("right_"):
                self.finger_ctrl[i] = -self.finger_ctrl[i]

        # QP 优化变量 = 6 个浮动基座自由度 + 29 个身体关节自由度
        self.ik_dof_cols = np.concatenate((np.arange(6), self.body_dof_adr)).astype(int)
        self.n_ik = len(self.ik_dof_cols)
        self.n_joints = len(BODY_JOINTS)
        self.ik_col = {int(d): i for i, d in enumerate(self.ik_dof_cols)}

        self.body_joint_lo = np.array([m.jnt_range[j, 0] for j in self.body_joint_ids])
        self.body_joint_hi = np.array([m.jnt_range[j, 1] for j in self.body_joint_ids])
        self.joint_limit_margin = np.full(self.n_joints, config.JOINT_LIMIT_MARGIN)
        for i, name in enumerate(BODY_JOINTS):
            if "wrist_pitch" in name or "wrist_yaw" in name:
                self.joint_limit_margin[i] = config.WRIST_LIMIT_MARGIN
        self.n_nonarm = len(LEG_JOINTS) + len(WAIST_JOINTS)

        # 足端接触球（用于支撑多边形）
        self.foot_contact_geoms = []
        for body_key in ("left_ankle", "right_ankle"):
            body_id = self.body_ids[body_key]
            for gid in range(m.ngeom):
                if (m.geom_bodyid[gid] == body_id
                        and m.geom_type[gid] == mujoco.mjtGeom.mjGEOM_SPHERE):
                    self.foot_contact_geoms.append(gid)

        self.jacp = np.zeros((3, m.nv))
        self.jacr = np.zeros((3, m.nv))

    # ------------------------------------------------------------------ #
    def reset_to_stand(self, settle_time):
        """直接构造一个微屈膝、质心位于支撑面中心的稳定站立姿态。

        直腿站立时膝关节处于奇异位形，垂直方向几乎不可控；屈膝 0.5 rad
        可让下蹲/重心调节变成良态问题，也更接近真人站立姿态。
        """
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        d.ctrl[:] = m.key_ctrl[0]
        d.ctrl[self.finger_act_ids] = self.finger_ctrl

        q = d.qpos.copy()
        for i, name in enumerate(FINGER_JOINTS):
            q[m.jnt_qposadr[self._id(mujoco.mjtObj.mjOBJ_JOINT, name)]] = \
                self.finger_ctrl[i]
        q[0], q[2] = -0.0215, 0.7699
        for side in (0, 6):
            q[self.body_qpos_adr[side + 0]] = -0.2523   # hip pitch
            q[self.body_qpos_adr[side + 3]] = 0.5215    # knee
            q[self.body_qpos_adr[side + 4]] = -0.2490   # ankle pitch
        q[self.body_qpos_adr[14]] = -0.0285             # waist pitch
        # 手臂初始外展：自然下垂时三指手拇指会插进大腿髋部网格约 32 mm，
        # 外展约 11° 后指尖与髋部留出 ~46 mm 间隙，即使伺服下垂也有 ~30 mm。
        q[self.body_qpos_adr[16]] = 0.40                # left shoulder roll
        q[self.body_qpos_adr[23]] = -0.40               # right shoulder roll
        d.qpos[:] = q
        d.qvel[:] = 0.0
        mujoco.mj_forward(m, d)

        self.q_target = d.qpos[self.body_qpos_adr].copy()
        # 用重力/偏置力矩前馈抵消位置伺服下垂，保持优化后的站姿
        kp = m.actuator_gainprm[self.body_act_ids, 0]
        tau = d.qfrc_bias[self.body_dof_adr]
        d.ctrl[self.body_act_ids] = self.q_target + np.clip(tau / kp, -0.1, 0.1)
        for _ in range(int(settle_time / m.opt.timestep)):
            mujoco.mj_step(m, d)
        self.q_target = d.qpos[self.body_qpos_adr].copy()

        self.hand_anchor = {
            key: (d.site_xpos[self.site_ids[key]].copy(), self.site_quat(key))
            for key in ("left_hand", "right_hand")
        }
        self.foot_anchor = {
            key: (d.site_xpos[self.site_ids[key]].copy(), self.site_quat(key))
            for key in ("left_foot", "right_foot")
        }
        self.torso_quat = d.xquat[self.body_ids["torso"]].copy()
        self.base_pos = d.qpos[:2].copy()
        self.base_quat = d.qpos[3:7].copy()
        self.com_target = d.subtree_com[0][:2].copy()

    def body_of_site(self, site_key):
        # site 所在的刚体：手 site 在 wrist_yaw_link 上，脚 site 在 ankle_roll_link 上
        if site_key == "left_hand":
            return self._id(mujoco.mjtObj.mjOBJ_BODY, "left_wrist_yaw_link")
        if site_key == "right_hand":
            return self._id(mujoco.mjtObj.mjOBJ_BODY, "right_wrist_yaw_link")
        if site_key == "left_foot":
            return self.body_ids["left_ankle"]
        if site_key == "right_foot":
            return self.body_ids["right_ankle"]
        raise KeyError(site_key)

    # ------------------------------------------------------------------ #
    def support_polygon(self, data=None):
        """左右脚接触球中心构成的凸包（支撑区域还需加上球半径）。"""
        d = self.data if data is None else data
        pts = [d.geom_xpos[gid][:2] for gid in self.foot_contact_geoms]
        return convex_hull_2d(pts)

    def support_margin(self, point, poly=None):
        """CoM 投影到支撑边界的有符号距离（已扣除接触球半径与安全裕度）。"""
        poly = self.support_polygon() if poly is None else poly
        normals, offsets = polygon_edges(poly)
        radius = float(self.model.geom_size[self.foot_contact_geoms[0], 0])
        return float(np.min(offsets - radius - config.SUPPORT_SHRINK
                            - normals @ np.asarray(point, float)))

    def site_quat(self, key):
        sid = self.site_ids[key]
        return mat_to_quat(self.data.site_xmat[sid].reshape(3, 3))

    def site_pose(self, key):
        sid = self.site_ids[key]
        return self.data.site_xpos[sid].copy(), self.site_quat(key)

    def head_pos(self):
        return self.data.geom_xpos[self.head_geom].copy()

    def head_quat(self):
        return mat_to_quat(self.data.geom_xmat[self.head_geom].reshape(3, 3))
