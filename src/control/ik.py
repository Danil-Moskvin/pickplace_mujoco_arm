import numpy as np
from utils.mj_utils import site_pos, jac_site_pos

def ik_step(model, data, q_des, dof_ids, target, site_name,
            lam=1e-2, step_gain=0.05, max_joint_vel=0.14):
    dt = float(model.opt.timestep)
    cur = site_pos(model, data, site_name)
    err = target - cur

    J = jac_site_pos(model, data, site_name)[:, dof_ids]
    dq = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(3), err)

    dq *= step_gain
    dq = np.clip(dq, -max_joint_vel * dt, max_joint_vel * dt)
    return q_des + dq

def move_scalar_towards(model, q_cur, q_target, max_vel):
    dt = float(model.opt.timestep)
    dq = np.clip(q_target - q_cur, -max_vel * dt, max_vel * dt)
    return q_cur + dq
