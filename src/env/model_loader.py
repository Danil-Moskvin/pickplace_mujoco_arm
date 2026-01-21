from pathlib import Path
import mujoco
from configs.settings import XML_FILENAME

def load_model():
    assets_dir = Path(__file__).resolve().parent / "assets"
    xml_path = assets_dir / XML_FILENAME
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    qpos_init = data.qpos.copy()
    qvel_init = data.qvel.copy()
    return model, data, qpos_init, qvel_init
