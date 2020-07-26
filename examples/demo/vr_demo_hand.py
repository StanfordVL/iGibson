import pybullet as p
import numpy as np
import time

from gibson2.core.physics.robot_locomotors import Fetch
from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, VrHand
from gibson2.core.simulator import Simulator
from gibson2.utils.utils import parse_config
from math import sqrt

optimize = True

# Timestep should always be set to 1/90 to match VR system's 90fps
s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# VR hand objects
lHand = VrHand()
s.import_articulated_object(lHand)
lHand.set_position([0.0, 0.0, 1.5])

rHand = VrHand()
s.import_articulated_object(rHand)
rHand.set_position([0.0, 0.0, 1.0])

# Load objects in the environment
for i in range(5):
    bottle = YCBObject('006_mustard_bottle')
    s.import_object(bottle)
    _, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
    bottle_pos = [1 ,0 - 0.2 * i, 1]
    p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

if optimize:
    s.optimize_data()

# Account for Gibson floors not being at z=0 - shift user height down by 0.2m
s.setVROffset([0, 0, -0.2])

# Runs simulation
while True:
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

    s.step(shouldTime=False)

    hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')

    if lIsValid:
        lHand.move_hand(lTrans, lRot)
        #lGripper.set_close_fraction(leftGripperFraction)

    if rIsValid:
        rHand.move_hand(rTrans, rRot)
        #rGripper.set_close_fraction(rightGripperFraction)

s.disconnect()