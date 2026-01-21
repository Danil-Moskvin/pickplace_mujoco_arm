import numpy as np
import mujoco

from configs import settings as S
from utils.mj_utils import (
    j_qpos, j_dof, site_pos, body_pos, aid
)
from control.ik import ik_step, move_scalar_towards
from control.gripper import gripper_open, gripper_pinch

class PickPlaceFSM:
    """
    FSM:
    0 above cube
    1 down to grasp
    1.5 pre-grasp pause
    2 close
    3 settle
    4 lift
    5 move above target
    6 down to place
    7 release + wait
    8 retreat up
    9 return home (same speed)
    10 done
    """
    def __init__(self, model, data, qpos_init, qvel_init):
        self.model = model
        self.data = data
        self.qpos_init = qpos_init
        self.qvel_init = qvel_init

        self.dof_ids = np.array([j_dof(model, j) for j in S.ARM_JOINTS_IK], dtype=int)

        self.paused = True
        self.stage = 0

        self.q_arm = S.HOME_Q.copy()
        self.q_wrist = float(S.HOME_WRIST)

        self.manual_grip = False
        self.manual_wrist_override = False

        self.pinch_s = 0.0
        self.settle_steps = 0
        self.pre_grasp_cnt = 0
        self.release_cnt = 0
        self.down_steps = 0

        self.target_above = None
        self.target_grasp = None
        self.target_lift = None
        self.target_move = None
        self.target_place = None
        self.target_retreat = None

        self.DOWN_TIMEOUT_STEPS = int(S.DOWN_TIMEOUT_SEC / model.opt.timestep)
        self.PRE_GRASP_STEPS = int(S.PRE_GRASP_PAUSE_SEC / model.opt.timestep)
        self.SETTLE_STEPS = int(S.SETTLE_SEC / model.opt.timestep)
        self.RELEASE_STEPS = int(S.RELEASE_SEC / model.opt.timestep)

        self.reset()

    def _recompute_pick_targets(self):
        cube_p = body_pos(self.model, self.data, S.BODY_CUBE)
        above = cube_p.copy()
        above[2] += S.ABOVE_Z

        grasp = cube_p.copy()
        grasp[2] = cube_p[2] + S.GRASP_Z_OFFSET
        return above, grasp

    def _make_lift_target(self):
        cur = site_pos(self.model, self.data, S.SITE_GRASP)
        t = cur.copy()
        t[2] += S.LIFT_DZ
        return t

    def _make_move_target(self):
        tsite = site_pos(self.model, self.data, S.SITE_TARGET)
        t = tsite.copy()
        t[2] = tsite[2] + S.CARRY_Z_ABOVE_TARGET
        return t

    def _make_place_target(self):
        tsite = site_pos(self.model, self.data, S.SITE_TARGET)
        t = tsite.copy()
        t[2] = tsite[2] + S.PLACE_Z_OFFSET
        return t

    def _make_retreat_target(self):
        cur = site_pos(self.model, self.data, S.SITE_GRASP)
        t = cur.copy()
        t[2] += S.RETREAT_DZ
        return t

    def reset(self):
        self.data.qpos[:] = self.qpos_init
        self.data.qvel[:] = self.qvel_init
        mujoco.mj_forward(self.model, self.data)

        self.data.qpos[j_qpos(self.model, "j1_yaw")] = float(S.HOME_Q[0])
        self.data.qpos[j_qpos(self.model, "j2_pitch")] = float(S.HOME_Q[1])
        self.data.qpos[j_qpos(self.model, "j3_pitch")] = float(S.HOME_Q[2])
        self.data.qpos[j_qpos(self.model, "j4_pitch")] = float(S.HOME_Q[3])
        self.data.qpos[j_qpos(self.model, S.WRIST_JOINT)] = float(S.HOME_WRIST)

        self.data.qpos[j_qpos(self.model, "j_finger_left")] = 0.0
        self.data.qpos[j_qpos(self.model, "j_finger_right")] = 0.04

        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.q_arm = S.HOME_Q.copy()
        self.q_wrist = float(S.HOME_WRIST)

        self.manual_grip = False
        self.manual_wrist_override = False

        self.stage = 0
        self.pinch_s = 0.0
        self.settle_steps = 0
        self.pre_grasp_cnt = 0
        self.release_cnt = 0
        self.down_steps = 0

        self.target_above, self.target_grasp = self._recompute_pick_targets()
        self.target_lift = None
        self.target_move = None
        self.target_place = None
        self.target_retreat = None

        print("RESET -> stage=0 (start paused). Press SPACE to run.")

    def on_key(self, keycode: int):
        try:
            ch = chr(keycode)
        except Exception:
            return

        if ch == ' ':
            self.paused = not self.paused
            if not self.paused:
                self.manual_grip = False
            print("⏸ PAUSED" if self.paused else "▶ RUN")

        elif ch in ('r', 'R'):
            print("RESET")
            self.reset()

        elif ch in ('g', 'G'):
            self.manual_grip = True
            if self.pinch_s < 1e-6:
                self.pinch_s = S.PINCH_TARGET
                print("G: CLOSE gripper")
            else:
                self.pinch_s = 0.0
                print("G: OPEN gripper")

        elif ch in ('q', 'Q'):
            self.manual_wrist_override = True
            self.q_wrist = float(np.clip(self.q_wrist + S.MANUAL_WRIST_STEP, -1.2, 1.2))
            print("Manual wrist:", self.q_wrist)

        elif ch in ('e', 'E'):
            self.manual_wrist_override = True
            self.q_wrist = float(np.clip(self.q_wrist - S.MANUAL_WRIST_STEP, -1.2, 1.2))
            print("Manual wrist:", self.q_wrist)

        elif ch in ('m', 'M'):
            self.manual_wrist_override = False
            print("Auto wrist DOWN enabled")

    def step(self):
        if not self.paused:
            if self.stage in (0, 1, 1.5, 2, 3, 4, 5, 6, 7, 8) and (not self.manual_wrist_override):
                self.q_wrist = move_scalar_towards(
                    self.model, self.q_wrist, S.WRIST_DOWN_ANGLE, S.MAX_JOINT_VEL_WRIST
                )

            if self.manual_grip:
                if self.pinch_s < 1e-6:
                    gripper_open(self.model, self.data, S.ACT_FL, S.ACT_FR)
                else:
                    gripper_pinch(self.model, self.data, self.pinch_s, S.ACT_FL, S.ACT_FR)
            else:
                if self.stage in (0, 1, 1.5):
                    gripper_open(self.model, self.data, S.ACT_FL, S.ACT_FR)
                elif self.stage in (2, 3, 4, 5, 6):
                    gripper_pinch(self.model, self.data, self.pinch_s, S.ACT_FL, S.ACT_FR)
                else:
                    gripper_open(self.model, self.data, S.ACT_FL, S.ACT_FR)

            if self.stage == 0:
                self.q_arm = ik_step(self.model, self.data, self.q_arm, self.dof_ids, self.target_above, S.SITE_ABOVE,
                                     lam=S.LAM, step_gain=S.STEP_ABOVE, max_joint_vel=S.MAX_JOINT_VEL_ARM)
                if np.linalg.norm(site_pos(self.model, self.data, S.SITE_ABOVE) - self.target_above) < S.TH_ABOVE:
                    self.stage = 1
                    self.down_steps = 0
                    print("stage=1 (down to cube)")

            elif self.stage == 1:
                self.down_steps += 1
                self.q_arm = ik_step(self.model, self.data, self.q_arm, self.dof_ids, self.target_grasp, S.SITE_GRASP,
                                     lam=S.LAM, step_gain=S.STEP_DOWN, max_joint_vel=S.MAX_JOINT_VEL_ARM)
                dist = np.linalg.norm(site_pos(self.model, self.data, S.SITE_GRASP) - self.target_grasp)
                if (dist < S.TH_GRASP) or (self.down_steps >= self.DOWN_TIMEOUT_STEPS):
                    self.stage = 1.5
                    self.pre_grasp_cnt = 0
                    print("stage=1.5 (pre-grasp pause)")

            elif self.stage == 1.5:
                self.pre_grasp_cnt += 1
                if self.pre_grasp_cnt >= self.PRE_GRASP_STEPS:
                    self.stage = 2
                    self.pinch_s = 0.0
                    self.settle_steps = 0
                    print("stage=2 (close)")

            elif self.stage == 2 and (not self.manual_grip):
                self.pinch_s = float(np.clip(self.pinch_s + S.PINCH_RATE * self.model.opt.timestep, 0.0, S.PINCH_TARGET))
                if abs(self.pinch_s - S.PINCH_TARGET) < 1e-6:
                    self.stage = 3
                    self.settle_steps = 0
                    print("stage=3 (settle)")

            elif self.stage == 3 and (not self.manual_grip):
                self.settle_steps += 1
                if self.settle_steps >= self.SETTLE_STEPS:
                    self.target_lift = self._make_lift_target()
                    self.stage = 4
                    print("stage=4 (lift)")

            elif self.stage == 4 and (not self.manual_grip):
                self.q_arm = ik_step(self.model, self.data, self.q_arm, self.dof_ids, self.target_lift, S.SITE_GRASP,
                                     lam=S.LAM, step_gain=S.STEP_LIFT, max_joint_vel=S.VEL_LIFT)
                if np.linalg.norm(site_pos(self.model, self.data, S.SITE_GRASP) - self.target_lift) < S.TH_LIFT:
                    self.target_move = self._make_move_target()
                    self.stage = 5
                    print("stage=5 (move above target)")

            elif self.stage == 5 and (not self.manual_grip):
                self.q_arm = ik_step(self.model, self.data, self.q_arm, self.dof_ids, self.target_move, S.SITE_GRASP,
                                     lam=S.LAM, step_gain=S.STEP_MOVE, max_joint_vel=S.VEL_MOVE)
                if np.linalg.norm(site_pos(self.model, self.data, S.SITE_GRASP) - self.target_move) < S.TH_MOVE:
                    self.target_place = self._make_place_target()
                    self.stage = 6
                    print("stage=6 (down to place)")

            elif self.stage == 6 and (not self.manual_grip):
                self.q_arm = ik_step(self.model, self.data, self.q_arm, self.dof_ids, self.target_place, S.SITE_GRASP,
                                     lam=S.LAM, step_gain=S.STEP_PLACE, max_joint_vel=S.VEL_PLACE)
                if np.linalg.norm(site_pos(self.model, self.data, S.SITE_GRASP) - self.target_place) < S.TH_PLACE:
                    self.stage = 7
                    self.release_cnt = 0
                    print("stage=7 (release)")

            elif self.stage == 7 and (not self.manual_grip):
                gripper_open(self.model, self.data, S.ACT_FL, S.ACT_FR)
                self.release_cnt += 1
                if self.release_cnt >= self.RELEASE_STEPS:
                    self.target_retreat = self._make_retreat_target()
                    self.stage = 8
                    print("stage=8 (retreat up)")

            elif self.stage == 8 and (not self.manual_grip):
                self.q_arm = ik_step(self.model, self.data, self.q_arm, self.dof_ids, self.target_retreat, S.SITE_GRASP,
                                     lam=S.LAM, step_gain=S.STEP_RETREAT, max_joint_vel=S.VEL_RETREAT)
                if np.linalg.norm(site_pos(self.model, self.data, S.SITE_GRASP) - self.target_retreat) < S.TH_RETREAT:
                    self.stage = 9
                    print("stage=9 (return home)")

            elif self.stage == 9 and (not self.manual_grip):
                dt = float(self.model.opt.timestep)
                dq_lim = S.HOME_RATE * dt
                self.q_arm = self.q_arm + np.clip(S.HOME_Q - self.q_arm, -dq_lim, dq_lim)

                if not self.manual_wrist_override:
                    self.q_wrist = move_scalar_towards(self.model, self.q_wrist, S.HOME_WRIST, S.MAX_JOINT_VEL_WRIST)

                if np.linalg.norm(self.q_arm - S.HOME_Q) < 0.03 and abs(self.q_wrist - S.HOME_WRIST) < 0.05:
                    self.stage = 10
                    print("DONE (home)")

        self.data.ctrl[aid(self.model, S.ACT_J1)] = float(self.q_arm[0])
        self.data.ctrl[aid(self.model, S.ACT_J2)] = float(self.q_arm[1])
        self.data.ctrl[aid(self.model, S.ACT_J3)] = float(self.q_arm[2])
        self.data.ctrl[aid(self.model, S.ACT_J4)] = float(self.q_arm[3])
        self.data.ctrl[aid(self.model, S.ACT_WRIST)] = float(self.q_wrist)
