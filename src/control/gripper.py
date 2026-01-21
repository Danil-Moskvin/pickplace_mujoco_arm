import numpy as np
from utils.mj_utils import aid


def gripper_open(model, data, act_fl="a_fL", act_fr="a_fR"):
    data.ctrl[aid(model, act_fl)] = 0.0
    data.ctrl[aid(model, act_fr)] = 0.04

def gripper_pinch(model, data, s, act_fl="a_fL", act_fr="a_fR"):
    s = float(np.clip(s, 0.0, 0.04))
    data.ctrl[aid(model, act_fl)] = -s
    data.ctrl[aid(model, act_fr)] = 0.04 - s
