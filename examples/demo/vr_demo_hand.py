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
s.setVROffset([1.0, 0, -0.2])

# Runs simulation
while True:
    s.step(shouldTime=False)

    hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')

    lTrig, lTouchX, lTouchY = s.getButtonDataForController('left_controller')
    rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

    # TODO: Add left hand back in for testing
    if rIsValid:
        rHand.move_hand(rTrans, rRot)
        rHand.toggle_finger_state(rTrig)

s.disconnect()