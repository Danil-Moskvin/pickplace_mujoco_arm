import numpy as np
import mujoco

def jid(model, name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
def aid(model, name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
def bid(model, name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
def sid(model, name): return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)

def j_qpos(model, joint_name) -> int:
    return int(model.jnt_qposadr[jid(model, joint_name)])

def j_dof(model, joint_name) -> int:
    return int(model.jnt_dofadr[jid(model, joint_name)])

def site_pos(model, data, site_name):
    return data.site_xpos[sid(model, site_name)].copy()

def body_pos(model, data, body_name):
    return data.xpos[bid(model, body_name)].copy()

def jac_site_pos(model, data, site_name):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, sid(model, site_name))
    return jacp
