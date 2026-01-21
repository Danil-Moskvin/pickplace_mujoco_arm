import sys
from pathlib import Path
import mujoco
import mujoco.viewer

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from env.model_loader import load_model
from control.fsm import PickPlaceFSM

def main():
    model, data, qpos_init, qvel_init = load_model()
    fsm = PickPlaceFSM(model, data, qpos_init, qvel_init)

    print("\nУправление:")
    print("  SPACE = pause/run")
    print("  R     = reset (robot + cube)")
    print("Кликни по окну Viewer, чтобы клавиатура работала.\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=fsm.on_key) as viewer:
        while viewer.is_running():
            fsm.step()
            mujoco.mj_step(model, data)
            viewer.sync()

if __name__ == "__main__":
    main()
