"""全局配置：路径、控制频率、QP 权重与回放参数。"""

from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]   # src/g1_wbc/config.py -> 工程根
ASSET_DIR = PKG_ROOT / "assets"
SCENE_XML = ASSET_DIR / "g1_demo_scene.xml"

SIM_TIMESTEP = 0.002          # 物理步长 500 Hz
VIDEO_FPS = 50
PRE_HOLD = 1.0                # 视频开头保持站立
POST_HOLD = 1.2               # 视频结尾保持

# ---- QP 任务权重 ----
W_HAND_POS = 600.0
W_HAND_ROT = 160.0
W_COM = 600.0
W_BASE_Z = 2000.0
W_TORSO = 120.0
# 姿态先验只用于约束零空间（保持自然站姿），不能和手部任务抢主导权：
# 旧值 15 + tau 0.2 s 会让抬手目标产生 3–8 cm 的稳态误差，并把参考状态
# 远远甩在目标后面，实际 CoM 与规划错位，是高速档向后倾倒的直接原因。
W_POSTURE = 3.0
W_DAMPING = 0.5

# ---- 任务时间常数（秒）：误差 -> 每步位姿增量 ----
TAU_HAND_POS = 0.06
TAU_HAND_ROT = 0.08
TAU_COM = 0.15
TAU_BASE_Z = 0.10
TAU_TORSO = 0.08
TAU_POSTURE = 0.50

# ---- 误差截断 ----
CLIP_HAND_POS = 0.10          # m
CLIP_HAND_ROT = 0.60          # rad
CLIP_COM = 0.05               # m

# ---- 关节/基座增量限幅 ----
MAX_JOINT_VEL = 1.0           # rad/s（腿/腰）
MAX_ARM_VEL = 2.0             # rad/s（肩/肘/腕，快速挥手时需要）
MAX_BASE_VEL = 0.20           # m/s
MAX_BASE_ANGVEL = 0.6         # rad/s
JOINT_LIMIT_MARGIN = 0.03     # rad
# 腕 pitch/yaw 在接近 ±1.61 rad 机械限位时，相邻腕连杆网格会互相压入约 0.5 mm
# （模型自带的网格重叠，不是动作导致的躯干穿模）。把 QP 的腕部限位余量加大，
# 避免解出落在重叠区的姿态。
WRIST_LIMIT_MARGIN = 0.09     # rad

# ---- 支撑多边形 ----
SUPPORT_SHRINK = 0.015        # m
SUPPORT_VEL_LOOKAHEAD = 0.2   # s，支撑约束中 CoM 速度的前瞻时间
SUPPORT_SLACK_N = 8
SUPPORT_SLACK_WEIGHT = 1.0e5

# ---- 回放：位置伺服参考增益与力矩前馈限幅 ----
REPLAY_K = 1.0
REPLAY_FF_LIMIT = 0.15        # rad

# CoM 目标相对支撑面中心的 x 偏置（负值向后，留出前向平衡裕度）
COM_BIAS_X = 0.0

# ---------------------------------------------------------------------------
# 动力学级 WBC（wbc_dyn.py）
# ---------------------------------------------------------------------------
# 任务权重
WBC_W_HAND_POS = 2.0e4
WBC_W_HAND_ROT = 1.0e5
# CoM 任务必须明显压过手部任务：否则手部为够目标会用屈膝/踝把 CoM 推出
# 支撑多边形（alternate/fast08 的慢速后倒就是这样发生的）。
WBC_W_COM = 1.0e6
# 双手前伸时把 CoM 参考向后预偏，抵消手臂质量前移与动态反作用
WBC_COM_FF_GAIN = 0.15
# 捕获点/CoM 距离支撑边界小于该值时开始放慢轨迹
WBC_GOV_MARGIN = 0.03
# 运动学回放模式的预测调速：裕度低于 KIN_GOV_MARGIN 时按比例降速。
# 注意：静止站立时 CoM 到支撑边界的裕度本身只有约 24 mm（已扣 SUPPORT_SHRINK）。
# 阈值取 0.03（静止时约 0.8 倍速），既能保留高速档的大部分速度，又能在
# CoM/捕获点接近后沿时及时减速；旧值 0.05 会让静止状态也只有 0.48 倍速。
KIN_GOV_MARGIN = 0.030
KIN_GOV_TAU_DOWN = 0.02   # 降速时间常数 [s]（危险时快速收紧）
KIN_GOV_TAU_UP = 0.10     # 恢复时间常数 [s]（缓慢放开）

# ---- 平衡前馈控制器（balance）----
# 原理：固定双脚时，手臂加速产生的反作用会移动 ZMP/CoP。让 CoM 目标沿
# 手臂反作用方向提前偏移（预期性姿势调整 APA），再用真实 CoM 速度阻尼
# 抑制倒立摆的慢发散；左右手加速度差驱动躯干反向转动，吸收角动量。
BAL_APA_GAIN = 0.6            # 1.0 ≈ 按双臂质量/总质量与 CoM 高度推导的理论值
BAL_APA_LEAD = 0.15           # s，目标加速度预览时间
BAL_VEL_DAMP = 0.25           # s，真实 CoM 速度反馈到 CoM 参考的阻尼
BAL_TORSO_DIFF_GAIN = 0.02    # rad/(m/s²)，左右手加速度差 → 躯干反向转动
BAL_ACC_TAU = 0.05            # s，手臂加速度一阶低通
BAL_VEL_DAMP_TAU = 0.06       # s，真实 CoM 速度一阶低通
BAL_COM_MARGIN = 0.008        # m，APA 目标距支撑边界的保留量
WBC_W_BASE_Z = 5.0e2
# 骨盆水平位置：固定双脚时骨盆只能在脚上方有限范围内移动。没有这个任务，
# 手部任务会通过屈膝/踝把骨盆整体向前“平移”出去，直到 CoM 越界。
WBC_W_BASE_XY = 2.0e4
WBC_KP_BASE_XY = 200.0
WBC_KD_BASE_XY = 20.0
WBC_A_BASE_XY_MAX = 10.0
WBC_W_TORSO = 3.0e3
# 头部/基座姿态稳定（头部是 torso 上的固定点，没有独立颈关节）：
# 头部前后晃动主要由基座平移 + 躯干俯仰组成，因此同时约束
# 头部水平位置和基座姿态，让 WBC 优先用踝/髋平移而不是躯干俯仰来平衡。
WBC_W_HEAD_POS = 1.5e4
WBC_KP_HEAD_POS = 400.0
WBC_KD_HEAD_POS = 40.0
WBC_A_HEAD_MAX = 25.0
WBC_HEAD_Y_SCALE = 0.5          # 侧向权重相对前后向的比例
# 无离线规划时（mocap/在线），任务本身躯干前倾超过该角度就淡出头部任务
WBC_HEAD_LEAN_GATE_DEG = 15.0
# 有规划时：直立任务把头部参考保持在初始位置（抑制前后晃），
# 任务本身需要弯腰时（pickup/shelf）按躯干前倾角平滑过渡到规划头部轨迹。
WBC_HEAD_FOLLOW_DEG = 20.0
# 直立任务对规划头部轨迹的跟随比例：0=完全锁在初始位置（最稳），
# 1=完全跟随规划（跟踪最好但头部前后位移大）。
WBC_HEAD_PLAN_FOLLOW = 0.0
WBC_W_BASE_ORIENT = 2.0e3
WBC_KP_BASE_ORIENT = 200.0
WBC_KD_BASE_ORIENT = 20.0
WBC_A_BASE_ORIENT_MAX = 40.0
WBC_W_FOOT_ORIENT = 1.0e3
WBC_KP_FOOT_ORIENT = 100.0
WBC_KD_FOOT_ORIENT = 15.0
WBC_A_FOOT_ORIENT_MAX = 30.0
WBC_W_FOOT_POS = 1.0e4
WBC_KP_FOOT_POS = 400.0
WBC_KD_FOOT_POS = 40.0
WBC_A_FOOT_POS_MAX = 30.0
WBC_W_POSTURE = 5.0e1
WBC_W_POSTURE_ARM = 2.0e1
WBC_W_DAMP = 2.0e1

# 正则项
WBC_W_QDD_REG = 1.0e-3
WBC_W_TAU_REG = 1.0e-3
WBC_W_FORCE_REG = 1.0
WBC_W_SLACK = 1.0e7

# 任务 PD 增益与限幅
WBC_KP_HAND_POS = 400.0
WBC_KD_HAND_POS = 40.0
WBC_CLIP_HAND_POS = 0.08
WBC_A_HAND_POS_MAX = 30.0
WBC_USE_HAND_JDDOT = True

WBC_KP_HAND_ROT = 800.0
WBC_KD_HAND_ROT = 80.0
WBC_CLIP_HAND_ROT = 0.60
WBC_A_HAND_ROT_MAX = 400.0

# DCM/倒立摆：ζ 提高以抑制固定双脚下最容易积累的慢发散；
# 加速度上限放宽到 6 m/s²，否则 1.4 m/s 的手速下平衡任务会先饱和。
WBC_OMEGA_LIP = 3.2
WBC_ZETA_LIP = 0.9
WBC_A_COM_MAX = 6.0
WBC_W_COM_Z = 1.0e3
WBC_KP_COM_Z = 200.0
WBC_KD_COM_Z = 20.0
WBC_A_COM_Z_MAX = 20.0
WBC_FZ_TOTAL_MIN_SCALE = 0.65
WBC_FZ_TOTAL_MAX_SCALE = 1.25

WBC_KP_BASE_Z = 400.0
WBC_KD_BASE_Z = 40.0

WBC_KP_TORSO = 100.0
WBC_KD_TORSO = 20.0
WBC_A_TORSO_MAX = 60.0

WBC_KP_POSTURE = 100.0
WBC_KD_POSTURE = 20.0
WBC_KP_POSTURE_ARM = 25.0
WBC_KD_POSTURE_ARM = 5.0
WBC_KD_JOINT_DAMP = 2.0

# 接触约束
# 支撑脚始终保留全部 8 个接触点（双脚各 4 个球）的力变量，不按瞬时 ncon
# 清零。否则脚跟一旦离地，该点的力被永久置零，CoP 无法回到脚跟，
# 卸载会自增强并直接翻倒。
WBC_STANCE_ALL_POINTS = True
WBC_USE_COP = True
WBC_MU_FOOT = 0.5
WBC_MU_YAW = 0.08
WBC_FZ_MIN = 50.0
WBC_FZ_MAX = 900.0
WBC_FZ_POINT_MIN = 0.0
WBC_FZ_POINT_MAX = 300.0
WBC_FXY_MAX = 400.0
WBC_MXY_MAX = 200.0
WBC_MZ_MAX = 80.0
WBC_COP_SHRINK = 0.010
WBC_SLACK_MAX = 5.0

# 加速度限幅与力矩
WBC_QDD_BASE_LIN_MAX = 30.0
WBC_QDD_BASE_ANG_MAX = 80.0
WBC_QDD_LEG_MAX = 200.0
WBC_QDD_ARM_MAX = 400.0
WBC_TAU_LIMIT_SCALE = 0.95
WBC_TAU_FALLBACK = 200.0
# 关节力矩阻尼会直接吃掉小关节（腕部 5 Nm）的有效力矩，造成明显跟踪滞后；
# QP 内已有 WBC_KD_JOINT_DAMP 阻尼任务，这里不再重复扣减。
WBC_TAU_DAMP = 0.0
