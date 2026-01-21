import numpy as np

# пути/имена
XML_FILENAME = "pickplace_arm.xml"

SITE_ABOVE = "ee_site"
SITE_GRASP = "grasp_site"

BODY_CUBE = "cube"
SITE_TARGET = "target_site"

# суставы
ARM_JOINTS_IK = ["j1_yaw", "j2_pitch", "j3_pitch", "j4_pitch"]
WRIST_JOINT = "j_wrist_pitch"

ACT_J1 = "a_j1"
ACT_J2 = "a_j2"
ACT_J3 = "a_j3"
ACT_J4 = "a_j4"
ACT_WRIST = "a_wrist"
ACT_FL = "a_fL"
ACT_FR = "a_fR"

# поза "домой"
HOME_Q = np.array([0.0, 0.6, -0.9, 0.4], dtype=float)
HOME_WRIST = 0.0

# кинематика/скорость
LAM = 1.0e-2

MAX_JOINT_VEL_ARM = 0.14       # рад/сек
MAX_JOINT_VEL_WRIST = 0.22     # рад/сек

STEP_ABOVE = 0.040
STEP_DOWN  = 0.050
STEP_LIFT  = 0.028
STEP_MOVE  = 0.035
STEP_PLACE = 0.030
STEP_RETREAT = STEP_MOVE

# ограничения скорости на отдельных этапах
VEL_LIFT   = 0.12
VEL_MOVE   = 0.12
VEL_PLACE  = 0.10
VEL_RETREAT = VEL_MOVE

# домой - той же скоростью, что и суставы
HOME_RATE = MAX_JOINT_VEL_ARM  # рад/сек

# геометрия
ABOVE_Z = 0.18
GRASP_Z_OFFSET = 0.04
LIFT_DZ = 0.14
CARRY_Z_ABOVE_TARGET = 0.18
PLACE_Z_OFFSET = 0.04
RETREAT_DZ = 0.20

# кисть
WRIST_DOWN_ANGLE = 0.7

# захват
PINCH_TARGET = 0.035
PINCH_RATE = 0.18
SETTLE_SEC = 0.25
RELEASE_SEC = 0.25

PRE_GRASP_PAUSE_SEC = 0.25

TH_ABOVE = 0.03
TH_GRASP = 0.02
TH_LIFT  = 0.02
TH_MOVE  = 0.03
TH_PLACE = 0.02
TH_RETREAT = 0.03

# таймаут опускания к кубику
DOWN_TIMEOUT_SEC = 4.0

MANUAL_WRIST_STEP = 0.10
