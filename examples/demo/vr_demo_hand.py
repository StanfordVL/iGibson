import pybullet as p
import numpy as np
import time

from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, VrHand
from gibson2.core.simulator import Simulator
from gibson2 import assets_path

model_path = assets_path + '\\models'
sample_urdf_folder = model_path + '\\sample_urdfs\\'

optimize = True

s = Simulator(mode='vr', timestep = 1/90.0, vrFullscreen=False, vrEyeTracking=False, optimized_renderer=optimize, vrMode=True)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# VR hand objects
lHand = VrHand(start_pos=[0.0, 0.0, 1.5])
s.import_articulated_object(lHand)

rHand = VrHand(start_pos=[0.0, 0.5, 1.5])
s.import_articulated_object(rHand)

# Object to pick up
model = InteractiveObj(sample_urdf_folder + 'object_H3ygj6efM8V.urdf')
s.import_object(model)
model.set_position([1,0,1])

if optimize:
    s.optimize_data()

# Start user close to counter for interaction
s.setVROffset([1.0, 0, -0.4])

while True:
    s.step(shouldPrintTime=False)

    hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')

    lTrig, lTouchX, lTouchY = s.getButtonDataForController('left_controller')
    rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

    # TODO: Get a left hand model!
    if rIsValid:
        rHand.move_hand(rTrans, rRot)
        rHand.toggle_finger_state(rTrig)

s.disconnect()