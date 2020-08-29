import pybullet as p
import numpy as np
import time

from gibson2.core.physics.robot_locomotors import Fetch
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, GripperObj, VisualMarker
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
s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=True, optimize_render=optimize, vrMode=vrMode)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# Grippers represent hands
lGripper = GripperObj()
s.import_articulated_object(lGripper)
lGripper.set_position([0.0, 0.0, 1.5])

rGripper = GripperObj()
s.import_articulated_object(rGripper)
rGripper.set_position([0.0, 0.0, 1.0])

# Load objects in the environment
for i in range(5):
    bottle = YCBObject('006_mustard_bottle')
    s.import_object(bottle)
    _, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
    bottle_pos = [1 ,0 - 0.2 * i, 1]
    p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

# Load eye gaze marker
gaze_marker = VisualMarker(radius=0.03)
s.import_object(gaze_marker)
gaze_marker.set_marker_pos([0,0,1.5])
marker_pos, _ = p.getBasePositionAndOrientation(gaze_marker.body_id)
print("Marker starting pos: ", marker_pos)

# Controls how closed each gripper is (maximally open to start)
leftGripperFraction = 0.0
rightGripperFraction = 0.0

if optimize:
    s.optimize_data()

# Control printing of fps data for rendering and physics
shouldTime = False
# Control printing of Anipal data with right trigger
shouldMoveMarker = False

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
                    shouldMoveMarker = True
                elif eventType == 'trigger_unpress':
                    rightGripperFraction = 0.0
                    shouldMoveMarker = False

    s.step(shouldPrintTime=shouldTime)

    if vrMode:
        hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
        lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
        rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')

        is_eye_data_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.getEyeTrackingData()

        if is_eye_data_valid and shouldMoveMarker:
            updated_marker_pos = [origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]]
            gaze_marker.set_marker_pos(updated_marker_pos)
            #print("Current hmd pos: ", hmdTrans)
            #print("Gaze origin in world space", origin)
            #print("Gaze dir", dir)
            #print("Updated marker pos: ", updated_marker_pos)
            #print("------------------")

        if lIsValid:
            lGripper.move_gripper(lTrans, lRot)
            lGripper.set_close_fraction(leftGripperFraction)

        if rIsValid:
            rGripper.move_gripper(rTrans, rRot)
            rGripper.set_close_fraction(rightGripperFraction)
            #print("Right controller x and y: ", rTrans[0], rTrans[1])
    
    elapsed = time.time() - start
    if (elapsed > 0):
        curr_fps = 1/elapsed
    else:
        curr_fps = 2000

    if shouldTime:
        print("Current fps: %f" % curr_fps)

s.disconnect()