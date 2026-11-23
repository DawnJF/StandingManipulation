"""内置双手 6-DOF 目标场景。

坐标约定：世界系，x 向前、y 向左、z 向上；机器人基座位于 x ≈ -0.037。
位置为手腕末端（掌心 site）目标，姿态为相对初始手姿态的世界系 RPY 偏移。
位置填 None 表示回到初始手位置；右手未显式给出时按左手沿 y 镜像。
"""

from dataclasses import dataclass
from typing import Optional, Tuple

from .fbx_motion import take_scenario

Row = Optional[Tuple[float, float, float]]
Rpy = Tuple[float, float, float]


@dataclass(frozen=True)
class Scenario:
    key: str
    title: str
    description: str
    times: Tuple[float, ...]  # 未缩放关键帧时间
    time_scale: float  # 时间缩放（>1 更慢）
    pre_hold: float
    post_hold: float
    left_pos: Tuple[Row, ...]
    left_rpy_deg: Tuple[Rpy, ...]
    right_pos: Optional[Tuple[Row, ...]] = None  # None = 左手 y 镜像
    right_rpy_deg: Optional[Tuple[Rpy, ...]] = None  # None = 左手 roll/yaw 取反
    base_dz: Tuple[float, ...] = ()  # 骨盆高度偏移，缺省全 0
    torso_pitch_deg: Tuple[float, ...] = ()  # 躯干世界系前倾角，缺省全 0（直立）
    com_bias_x: float = 0.0  # CoM 目标前后偏置（负值向后）
    interp: str = "cubic"  # 位置插值：cubic 或 minjerk
    support_lookahead: float = 0.2  # 支撑约束的 CoM 速度前瞻 [s]
    leg_stiffness_scale: float = 1.0  # 腿部位置伺服刚度倍率
    arm_vel_limit: float = 2.0  # 臂关节速度限幅 [rad/s]
    base_vel_limit: float = 0.20  # 基座平移速度限幅 [m/s]
    base_angvel_limit: float = 0.6  # 基座角速度限幅 [rad/s]
    replay_k: float = 1.0  # 回放位置伺服附加增益
    reach_lateral: float = 0.0  # 首/末段横向外让量 [m]

    @property
    def duration(self) -> float:
        return float(self.times[-1]) * self.time_scale

    @property
    def total_duration(self) -> float:
        return self.pre_hold + self.duration + self.post_hold


def _fast_scenario(
    key, title, description, segment, pos_a, pos_b, rpy_a, rpy_b, crouch=-0.010
):
    """构造“来到起点 -> 往返 2 个来回 -> 回位”的高速摆动场景。

    使用 C2 三次样条（折返点速度连续），避免 min-jerk 在每个关键帧
    速度归零造成的反复制动冲击；segment 仍用于区分 0.6/0.8/1.0 三档。

    注意：三次样条在到达第一个目标点前会沿 y 向内凹（因为该点之后 y 要
    反向），端点越靠近身体中线，内凹越深。reach_lateral 给首/末两段叠加一个
    平滑的横向外让（两端为零、导数连续），让手先离开身体再内收。
    """
    reach, ret = 1.8, 2.0
    t1 = reach
    t2 = t1 + segment
    t3 = t2 + segment
    t4 = t3 + segment
    t5 = t4 + segment
    t6 = t5 + ret
    return Scenario(
        key=key,
        title=title,
        description=description,
        times=(0.0, t1, t2, t3, t4, t5, t6),
        time_scale=1.0,
        pre_hold=0.8,
        post_hold=0.8,
        interp="cubic",
        leg_stiffness_scale=1.5,
        arm_vel_limit=8.0,  # G1 臂关节实际能力远高于 2 rad/s，先放开以免人为卡死
        replay_k=1.0,  # 平滑轨迹 + 自适应调速后不再需要 8 倍刚度
        reach_lateral=0.12,  # 首/末段手先横向让开约 12 cm，避免拇指扫到髋部
        left_pos=(None, pos_a, pos_b, pos_a, pos_b, pos_a, None),
        left_rpy_deg=((0, 0, 0), rpy_a, rpy_b, rpy_a, rpy_b, rpy_a, (0, 0, 0)),
        base_dz=(0.0, crouch, crouch, crouch, crouch, crouch, 0.0),
    )


# 对角往返路径：|Δp| = 0.230 m，峰值手速 = 1.875 * |Δp| / segment
# 手位必须落在“不做大幅前倾也能到”的可达范围内：目标越靠前，全身 QP 越会
# 用前移骨盆去凑手位，CoM 被推到支撑边界（规划裕度 ~0），实机随即失稳。
_SWING_A = (0.26, 0.24, 0.92)
_SWING_B = (0.34, 0.36, 1.10)
_SWING_RPY_A = (0, -60, 0)
_SWING_RPY_B = (0, -75, 0)

# 前后推拉路径：|Δp| = 0.200 m
_PUSH_A = (0.21, 0.24, 0.89)
_PUSH_B = (0.37, 0.24, 1.01)


SCENARIOS = {
    # ------------------------------------------------------------------ #
    "demo": Scenario(
        key="demo",
        title="默认综合动作",
        description="抬手、前伸、下探、回收，腰腿随动并轻度下蹲；"
        "动作 12 s、完整执行约 16.3 s，峰值手速约 0.33 m/s"
        "（固定双脚下的稳定上限）。",
        times=(0.0, 2.5, 5.0, 7.5, 10.0, 12.0),
        time_scale=0.2,
        pre_hold=0.5,
        post_hold=0.6,
        left_pos=(
            None,
            (0.28, 0.20, 0.94),
            (0.26, 0.26, 1.10),
            (0.26, 0.18, 0.84),
            (0.30, 0.22, 0.95),
            None,
        ),
        left_rpy_deg=(
            (0, 0, 0),
            (0, -90, 0),
            (0, -80, 8),
            (0, -95, 0),
            (5, -90, -5),
            (0, 0, 0),
        ),
        base_dz=(0.0, 0.0, 0.005, -0.022, 0.0, 0.0),
    ),
    # ------------------------------------------------------------------ #
    "wave": Scenario(
        key="wave",
        title="快速挥手",
        description="先向前伸手，再在胸腹高度左右挥动 3 个来回（约 0.7 Hz），"
        "腕部 yaw ±30°；动作 7.7 s、完整执行约 9.0 s，峰值手速约 0.42 m/s。",
        times=(0.0, 2.8, 4.0, 5.2, 6.4, 7.6, 8.8, 10.0, 12.8),
        time_scale=0.2,
        pre_hold=0.5,
        post_hold=0.6,
        interp="cubic",
        leg_stiffness_scale=3.0,
        reach_lateral=0.10,
        left_pos=(
            None,
            (0.28, 0.20, 0.94),
            (0.28, 0.16, 0.97),
            (0.28, 0.30, 0.93),
            (0.28, 0.16, 0.97),
            (0.28, 0.30, 0.93),
            (0.28, 0.20, 0.95),
            (0.29, 0.21, 0.95),
            None,
        ),
        left_rpy_deg=(
            (0, 0, 0),
            (0, -85, 10),
            (0, -75, 30),
            (0, -75, -30),
            (0, -75, 30),
            (0, -75, -30),
            (0, -80, 10),
            (0, -70, 0),
            (0, 0, 0),
        ),
        base_dz=(0.0, -0.010, -0.005, -0.005, -0.005, -0.005, -0.010, -0.005, 0.0),
    ),
    # ------------------------------------------------------------------ #
    "alternate": Scenario(
        key="alternate",
        title="左右交替高低",
        description="左手抬高、右手下探后互换，单臂垂直行程 0.22 m，"
        "约 1.4 s 完成一次交换，完整执行约 11.6 s，重心随手臂左右转移。",
        times=(0.0, 2.8, 4.2, 6.6, 8.9, 10.2, 12.9),
        time_scale=0.3,
        pre_hold=0.5,
        post_hold=0.6,
        interp="cubic",
        leg_stiffness_scale=3.0,
        left_pos=(
            None,
            (0.28, 0.20, 0.94),
            (0.27, 0.22, 1.08),
            (0.27, 0.18, 0.86),
            (0.28, 0.20, 0.94),
            (0.29, 0.21, 0.95),
            None,
        ),
        left_rpy_deg=(
            (0, 0, 0),
            (0, -85, 0),
            (0, -60, 0),
            (0, 10, 0),
            (0, -50, 0),
            (0, -40, 0),
            (0, 0, 0),
        ),
        right_pos=(
            None,
            (0.28, -0.20, 0.94),
            (0.27, -0.18, 0.86),
            (0.27, -0.22, 1.08),
            (0.28, -0.20, 0.94),
            (0.29, -0.21, 0.95),
            None,
        ),
        right_rpy_deg=(
            (0, 0, 0),
            (0, -85, 0),
            (0, 10, 0),
            (0, -60, 0),
            (0, -50, 0),
            (0, -40, 0),
            (0, 0, 0),
        ),
        base_dz=(0.0, -0.008, -0.006, -0.010, -0.006, -0.004, 0.0),
    ),
    # ------------------------------------------------------------------ #
    "circle": Scenario(
        key="circle",
        title="双手反相画圆",
        description="双手在身前各画半径 0.08 m 的圆（约 0.9 Hz），左右相位相差 180°，"
        "腕部随圆周转向；完整执行约 7.0 s。",
        times=(0.0, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3, 7.0, 7.7, 8.4, 9.6, 12.4),
        time_scale=0.2,
        pre_hold=0.5,
        post_hold=0.6,
        interp="cubic",
        leg_stiffness_scale=3.0,
        reach_lateral=0.12,
        left_pos=(
            None,
            (0.28, 0.20, 0.94),
            (0.26, 0.21, 1.02),
            (0.26, 0.168, 1.002),
            (0.26, 0.15, 0.96),
            (0.26, 0.168, 0.918),
            (0.26, 0.21, 0.90),
            (0.26, 0.252, 0.918),
            (0.26, 0.27, 0.96),
            (0.26, 0.252, 1.002),
            (0.27, 0.21, 0.96),
            None,
        ),
        left_rpy_deg=(
            (0, 0, 0),
            (0, -85, 10),
            (0, -75, 20),
            (0, -75, 0),
            (0, -75, -20),
            (0, -75, 0),
            (0, -75, 20),
            (0, -75, 0),
            (0, -75, -20),
            (0, -75, 0),
            (0, -65, 0),
            (0, 0, 0),
        ),
        right_pos=(
            None,
            (0.28, -0.20, 0.94),
            (0.26, -0.21, 0.90),
            (0.26, -0.168, 0.918),
            (0.26, -0.15, 0.96),
            (0.26, -0.168, 1.002),
            (0.26, -0.21, 1.02),
            (0.26, -0.252, 1.002),
            (0.26, -0.27, 0.96),
            (0.26, -0.252, 0.918),
            (0.27, -0.21, 0.96),
            None,
        ),
        base_dz=(
            0.0,
            -0.008,
            -0.004,
            -0.008,
            -0.004,
            -0.008,
            -0.004,
            -0.008,
            -0.004,
            -0.008,
            -0.004,
            0.0,
        ),
    ),
    # ------------------------------------------------------------------ #
    "converge": Scenario(
        key="converge",
        title="向中线收拢 + 大幅腕转",
        description="双手向身体中线收拢至相距 30 cm，腕部 yaw 45°、pitch -65°，再展开回位；"
        "约 0.7 s 完成一次收拢/展开，完整执行约 7.3 s。",
        times=(0.0, 2.8, 4.6, 6.4, 8.2, 9.6, 10.8, 13.6),
        time_scale=0.2,
        pre_hold=0.5,
        post_hold=0.6,
        interp="cubic",
        leg_stiffness_scale=3.0,
        reach_lateral=0.10,
        left_pos=(
            None,
            (0.28, 0.20, 0.94),
            (0.27, 0.15, 0.97),
            (0.27, 0.15, 1.00),
            (0.28, 0.18, 0.95),
            (0.29, 0.20, 0.95),
            (0.29, 0.21, 0.95),
            None,
        ),
        left_rpy_deg=(
            (0, 0, 0),
            (0, -85, 20),
            (0, -65, 45),
            (0, -65, 45),
            (0, -50, 20),
            (0, -70, 0),
            (0, -60, 0),
            (0, 0, 0),
        ),
        base_dz=(0.0, -0.008, -0.012, -0.014, -0.010, -0.006, -0.004, 0.0),
    ),
    # ------------------------------------------------------------------ #
    # 真实任务动作：关键帧与时间取自人体日常动作的运动学特征（见 README）。
    # 所有动作双脚固定，通过髋膝踝下蹲 + 躯干前倾完成，手臂不参与接触。
    # ------------------------------------------------------------------ #
    "pickup": Scenario(
        key="pickup",
        title="弯腰捡起地面物品",
        description="屈膝下蹲 + 躯干前倾约 50°，左手下探到大腿外侧"
        "（腕高 0.32 m，指尖约 0.22 m）抓取 15 cm 高的物品，"
        "随后起身提到胸前；不要求触地（固定双脚 + 不迈步时下方可达"
        "受限，且手贴大腿会穿模）。右手轻扶右腿，完整执行约 6.5 s。",
        times=(0.0, 1.4, 2.4, 3.0, 4.4, 5.4, 6.4),
        time_scale=0.266667,
        pre_hold=0.5,
        post_hold=0.6,
        interp="cubic",
        leg_stiffness_scale=3.0,
        arm_vel_limit=5.0,
        base_vel_limit=0.30,
        base_angvel_limit=0.9,
        left_pos=(
            None,
            (0.18, 0.30, 0.50),
            (0.24, 0.30, 0.32),
            (0.24, 0.30, 0.32),
            (0.24, 0.26, 0.74),
            (0.22, 0.18, 0.88),
            None,
        ),
        left_rpy_deg=(
            (0, 0, 0),
            (0, -20, 0),
            (0, 0, 0),
            (0, 0, 0),
            (0, -40, 0),
            (0, -70, 0),
            (0, 0, 0),
        ),
        right_pos=(
            None,
            (0.05, -0.27, 0.62),
            (0.07, -0.27, 0.56),
            (0.07, -0.27, 0.56),
            (0.04, -0.27, 0.62),
            (0.03, -0.28, 0.64),
            None,
        ),
        right_rpy_deg=(
            (0, 0, 0),
            (0, -40, 0),
            (0, -50, 0),
            (0, -50, 0),
            (0, -40, 0),
            (0, -30, 0),
            (0, 0, 0),
        ),
        base_dz=(0.0, -0.14, -0.32, -0.32, -0.08, 0.0, 0.0),
        torso_pitch_deg=(0.0, 27.0, 50.0, 50.0, 18.0, 2.0, 2.0),
    ),
    # ------------------------------------------------------------------ #
    "shelf": Scenario(
        key="shelf",
        title="弯腰从矮柜取物",
        description="躯干前倾约 38°、微蹲，右手伸入 0.58 m 高的矮柜层板抓取物品，"
        "再直腰把物品收到胸前；在线调速后完整执行约 8.3 s。",
        times=(0.0, 1.1, 1.9, 2.4, 3.4, 4.2, 5.0),
        time_scale=0.316667,
        pre_hold=0.5,
        post_hold=0.6,
        interp="cubic",
        leg_stiffness_scale=3.0,
        arm_vel_limit=4.0,
        base_vel_limit=0.25,
        base_angvel_limit=0.8,
        left_pos=(
            None,
            (0.00, 0.30, 0.71),
            (0.02, 0.30, 0.69),
            (0.02, 0.30, 0.69),
            (0.00, 0.30, 0.71),
            (0.00, 0.30, 0.71),
            None,
        ),
        left_rpy_deg=(
            (0, 0, 0),
            (0, -20, 0),
            (0, -25, 0),
            (0, -25, 0),
            (0, -20, 0),
            (0, -20, 0),
            (0, 0, 0),
        ),
        right_pos=(
            None,
            (0.28, -0.16, 0.82),
            (0.40, -0.15, 0.58),
            (0.40, -0.15, 0.58),
            (0.26, -0.14, 0.78),
            (0.22, -0.15, 0.90),
            None,
        ),
        right_rpy_deg=(
            (0, 0, 0),
            (0, -60, 0),
            (-15, -85, 0),
            (-15, -85, 0),
            (0, -80, 0),
            (0, -85, 0),
            (0, 0, 0),
        ),
        base_dz=(0.0, -0.04, -0.14, -0.14, -0.06, 0.0, 0.0),
        torso_pitch_deg=(0.0, 18.0, 38.0, 38.0, 20.0, 2.0, 2.0),
    ),
    # ------------------------------------------------------------------ #
    "greet": Scenario(
        key="greet",
        title="抬手打招呼",
        description="右手抬到脸侧 1.30 m、掌心朝前，以约 1.1 Hz 左右摆动 3 个来回，"
        "再放下；左手自然下垂，躯干保持直立，在线调速后约 7.7 s。",
        times=(0.0, 1.0, 1.6, 2.05, 2.50, 2.95, 3.40, 3.85, 4.6, 5.4),
        time_scale=0.333333,
        pre_hold=0.5,
        post_hold=0.6,
        interp="cubic",
        leg_stiffness_scale=3.0,
        arm_vel_limit=4.0,
        base_vel_limit=0.20,
        base_angvel_limit=0.6,
        left_pos=(None,) * 10,
        left_rpy_deg=((0, 0, 0),) * 10,
        right_pos=(
            None,
            (0.10, -0.22, 1.12),
            (0.10, -0.28, 1.30),
            (0.10, -0.24, 1.30),
            (0.10, -0.34, 1.30),
            (0.10, -0.24, 1.30),
            (0.10, -0.34, 1.30),
            (0.10, -0.24, 1.30),
            (0.02, -0.24, 0.85),
            None,
        ),
        right_rpy_deg=(
            (0, 0, 0),
            (0, -100, 0),
            (0, -110, 10),
            (0, -110, -15),
            (0, -110, 15),
            (0, -110, -15),
            (0, -110, 15),
            (0, -110, -15),
            (0, -90, 0),
            (0, 0, 0),
        ),
        base_dz=(
            0.0,
            -0.004,
            -0.008,
            -0.006,
            -0.008,
            -0.006,
            -0.008,
            -0.006,
            -0.004,
            0.0,
        ),
        torso_pitch_deg=(0.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 1.0, 0.0),
    ),
    # ------------------------------------------------------------------ #
    "open_door": Scenario(
        key="open_door",
        title="拉开柜门",
        description="右手前伸到 1.06 m 高的柜门把手，腕部转动约 40°，再沿圆弧把门拉开 "
        "0.30 m；单手动作，躯干轻微前倾，在线调速后约 5.7 s。",
        times=(0.0, 1.0, 1.5, 2.7, 3.2, 4.2),
        time_scale=0.333333,
        pre_hold=0.5,
        post_hold=0.6,
        interp="cubic",
        leg_stiffness_scale=3.0,
        arm_vel_limit=4.0,
        base_vel_limit=0.20,
        base_angvel_limit=0.6,
        left_pos=(None,) * 6,
        left_rpy_deg=((0, 0, 0),) * 6,
        right_pos=(
            None,
            (0.30, -0.20, 1.06),
            (0.30, -0.20, 1.06),
            (0.24, -0.34, 1.03),
            (0.24, -0.34, 1.03),
            None,
        ),
        right_rpy_deg=(
            (0, 0, 0),
            (0, -90, 0),
            (-40, -90, 0),
            (-40, -90, -25),
            (-40, -90, -25),
            (0, 0, 0),
        ),
        base_dz=(0.0, -0.010, -0.014, -0.020, -0.018, 0.0),
        torso_pitch_deg=(0.0, 6.0, 8.0, 10.0, 9.0, 0.0),
    ),
    # ------------------------------------------------------------------ #
    # 高速档：同一几何路径，单段时长不同 -> 目标峰值手速 0.6 / 0.8 / 1.0 m/s
    "fast06": _fast_scenario(
        "fast06",
        "高速档 0.6 m/s",
        "双手沿对角线往返 2 个来回，目标峰值手速 ≈ 0.55 m/s（约为此前最快场景的 2 倍）。",
        segment=0.820,
        pos_a=_SWING_A,
        pos_b=_SWING_B,
        rpy_a=_SWING_RPY_A,
        rpy_b=_SWING_RPY_B,
    ),
    "fast08": _fast_scenario(
        "fast08",
        "高速档 0.8 m/s",
        "同一对角线路径，目标峰值手速 ≈ 0.73 m/s。",
        segment=0.615,
        pos_a=_SWING_A,
        pos_b=_SWING_B,
        rpy_a=_SWING_RPY_A,
        rpy_b=_SWING_RPY_B,
    ),
    "fast10": _fast_scenario(
        "fast10",
        "高速档 1.0 m/s",
        "同一对角线路径，目标峰值手速 ≈ 0.91 m/s，用于观察失败边界。",
        segment=0.492,
        pos_a=_SWING_A,
        pos_b=_SWING_B,
        rpy_a=_SWING_RPY_A,
        rpy_b=_SWING_RPY_B,
    ),
    "fast10_dir": _fast_scenario(
        "fast10_dir",
        "1.0 m/s 前后推拉",
        "双手同向（一起前推/后拉）而非左右镜像，目标峰值手速 ≈ 1.0 m/s，直接冲击前后平衡。",
        segment=0.375,
        pos_a=_PUSH_A,
        pos_b=_PUSH_B,
        rpy_a=(0, -80, 0),
        rpy_b=(0, -65, 0),
    ),
    # ------------------------------------------------------------------ #
    # FBX 动捕重定向（data/fbx，逻辑见 fbx_motion.py，数据缓存在 data/retarget）
    "take01": take_scenario("take01"),
    "take02": take_scenario("take02"),
    "take03": take_scenario("take03"),
}


def get_scenario(key: str) -> Scenario:
    try:
        return SCENARIOS[key]
    except KeyError:
        raise KeyError(f"未知场景 {key!r}，可选：{', '.join(SCENARIOS)}") from None


def describe_scenarios() -> str:
    lines = []
    for spec in SCENARIOS.values():
        lines.append(
            f"- {spec.key:10s} {spec.title}（{spec.total_duration:.1f} s）"
            f"\n    {spec.description}"
        )
    return "\n".join(lines)
