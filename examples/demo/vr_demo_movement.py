import pybullet as p
import numpy as np
import time

from gibson2.core.physics.robot_locomotors import Fetch
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, GripperObj
from gibson2.core.simulator import Simulator
from gibson2 import assets_path, dataset_path
from gibson2.utils.utils import parse_config

model_path = assets_path + '\\models\\'
gripper_folder = model_path + '\\gripper\\'
configs_folder = '..\\configs\\'
fetch_config = parse_config(configs_folder + 'fetch_p2p_nav.yaml')

optimize = True
# Toggle this to only use renderer without VR, for testing purposes
vrMode = True

# Timestep should always be set to 1/90 to match VR system's 90fps
s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize, vrMode=vrMode)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# Fetch robot in scene
#fetch = Fetch(fetch_config)
#s.import_robot(fetch)
#fetch.set_position([0,0,0])
#fetch.robot_specific_reset()

# Grippers represent hands
lGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(lGripper)
lGripper.set_position([0.0, 0.0, 1.5])

rGripper = GripperObj(gripper_folder + 'gripper.urdf')
s.import_articulated_object(rGripper)
rGripper.set_position([0.0, 0.0, 1.0])

# Load objects in the environment
for i in range(5):
    bottle = YCBObject('006_mustard_bottle')
    s.import_object(bottle)
    _, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
    bottle_pos = [1 ,0 - 0.2 * i, 1]
    p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

# Controls how closed each gripper is (maximally open to start)
leftGripperFraction = 0.0
rightGripperFraction = 0.0

if optimize:
    s.optimize_data()

# Account for Gibson floors not being at z=0 - shift user height down by 0.2m
s.setVROffset([0, 0, -0.2])

# Runs simulation
while True:
    start = time.time()

    if vrMode:
        eventList = s.pollVREvents()
        for event in eventList:
            deviceType, eventType = event
            if deviceType == 'left_controller':
                if eventType == 'trigger_press':
                    leftGripperFraction = 0.8
                elif eventType == 'trigger_unpress':
                    leftGripperFraction = 0.0
            elif deviceType == 'right_controller':
                if eventType == 'trigger_press':
                    rightGripperFraction = 0.8
                elif eventType == 'trigger_unpress':
                    rightGripperFraction = 0.0

    s.step(shouldTime=True)

    if vrMode:
        hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
        lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
        rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')
        # TODO: Make nice interface functions for this function
        lTrig, lTouchX, lTouchY = s.getButtonDataForController('left_controller')
        rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

        # Uncomment to see/debug analog data
        #print("Printing trigger and touch data for left then right:")
        #if lIsValid:
        #    print("Left: ", lTrig, lTouchX, lTouchY)
        #if rIsValid:
        #    print("Right: ", rTrig, rTouchX, rTouchY)

        if lIsValid:
            lGripper.move_gripper(lTrans, lRot)
            lGripper.set_close_fraction(leftGripperFraction)

        if rIsValid:
            rGripper.move_gripper(rTrans, rRot)
            rGripper.set_close_fraction(rightGripperFraction)

        current_offset = s.getVROffset()

        # Move the VR player in the direction of the analog stick
        # In this implementation, +ve x and +ve y correspond to the same axes in Gibson
        # Only uses data from right controller
        # TODO: Implement a system where movement is relative to direction of HMD?
        if rIsValid:
            # Small offsets since this method could be call 100 times a second
            rTouchXOffset = rTouchX * 0.003
            rTouchYOffset = rTouchY * 0.003
            s.setVROffset([current_offset[0] + rTouchXOffset, current_offset[1] + rTouchYOffset, current_offset[2]])
    
    elapsed = time.time() - start
    if (elapsed > 0):
        curr_fps = 1/elapsed
    else:
        curr_fps = 2000

    print("Current fps: %f" % curr_fps)

s.disconnect()