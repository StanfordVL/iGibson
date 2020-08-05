import pybullet as p
import numpy as np
import time

from gibson2.core.physics.robot_locomotors import Fetch
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, GripperObj
from gibson2.core.simulator import Simulator
from gibson2 import assets_path, dataset_path
from gibson2.utils.utils import parse_config
from gibson2.utils.vr_utils import get_normalized_translation_vec, translate_vr_position_by_vecs
from math import sqrt

model_path = assets_path + '\\models\\'
gripper_folder = model_path + '\\gripper\\'
configs_folder = '..\\configs\\'
fetch_config = parse_config(configs_folder + 'fetch_p2p_nav.yaml')

optimize = True
# Toggle this to only use renderer without VR, for testing purposes
vrMode = True
# Possible types: hmd_relative, torso_relative
movement_type = 'torso_relative'

# Timestep should always be set to 1/90 to match VR system's 90fps
s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize, vrMode=vrMode)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# User controls fetch in this demo
fetch = Fetch(fetch_config, vr_mode=True)
s.import_robot(fetch)
fetch.set_position([-1.5,0,0])
fetch.robot_specific_reset()

# Load objects in the environment
for i in range(5):
    bottle = YCBObject('006_mustard_bottle')
    s.import_object(bottle)
    _, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
    bottle_pos = [1 ,0 - 0.2 * i, 1]
    p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

if optimize:
    s.optimize_data()

fetch_height = 1.2

effector_start_pos = fetch.get_end_effector_position()
# TODO: This is not quite working yet
#fetch.create_end_effector_constraint(effector_start_pos)

while True:
    s.step(shouldTime=False)

    if vrMode:
        hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
        lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
        rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')
        lTrig, lTouchX, lTouchY = s.getButtonDataForController('left_controller')
        rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

        # Only use z angle to rotate fetch around vertical axis
        _, _, hmd_z = p.getEulerFromQuaternion(hmdRot)
        fetch_rot = p.getQuaternionFromEuler([0, 0, hmd_z])
        fetch.set_orientation(fetch_rot)

        hmd_world_pos = s.getHmdWorldPos()
        fetch_pos = fetch.get_position()

        # Calculate x and y offset to get to fetch position
        # z offset is to the desired hmd height, corresponding to fetch head height
        offset_to_fetch = [fetch_pos[0] - hmd_world_pos[0], 
                        fetch_pos[1] - hmd_world_pos[1], 
                        fetch_height - hmd_world_pos[2]] 

        s.setVROffset(offset_to_fetch)

        relative_device = 'hmd'
        right, up, forward = s.getDeviceCoordinateSystem(relative_device)

        # Move the VR player in the direction of the analog stick
        # In this implementation, +ve x corresponds to right and +ve y corresponds to forward
        # relative to the HMD
        # Only uses data from right controller
        if rIsValid:
            new_fetch_position = translate_vr_position_by_vecs(rTouchX, rTouchY, right, forward, fetch.get_position(), 0.005)
            fetch.set_position(new_fetch_position)
            # TODO: This is not quite working yet
            #fetch.change_movement_constraint(rTrans, rRot)

s.disconnect()