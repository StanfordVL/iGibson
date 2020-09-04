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

s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# VR hand objects
lHand = VrHand(start_pos=[0.0, 0.0, 1.5])
s.import_articulated_object(lHand)

rHand = VrHand(start_pos=[0.0, 0.5, 1.5])
s.import_articulated_object(rHand)

# Object to pick up
# model = InteractiveObj(sample_urdf_folder + 'object_H3ygj6efM8V.urdf')
# s.import_object(model)
# model.set_position([1,0,1])

# for i in range(5):
#     bottle = YCBObject('006_mustard_bottle')
#     s.import_object(bottle)
#     _, org_orn = p.getBasePositionAndOrientation(bottle.body_id)
#     bottle_pos = [1, 0 - 0.2 * i, 1]
#     p.resetBasePositionAndOrientation(bottle.body_id, bottle_pos, org_orn)

fname = r'C:\Users\igibs\iGibson\gibson2\assets\dataset\processed\canned_food\2\rigid_body.urdf'
scale = 0.6071
object_pos = [1, 0, 1]
object_orn = [0.707, 0.707, 0, 0]
obj1 = InteractiveObj(filename=fname, scale=scale)
s.import_object(obj1)
obj1.set_position_orientation(object_pos, object_orn)

fname = r'C:\Users\igibs\iGibson\gibson2\assets\dataset\processed\canned_food\7\rigid_body.urdf'
scale = 0.6764
object_pos = [1, -0.2, 1]
object_orn = [0.707, 0.707, 0, 0]
obj2 = InteractiveObj(filename=fname, scale=scale)
s.import_object(obj2)
obj2.set_position_orientation(object_pos, object_orn)

fname = r'C:\Users\igibs\iGibson\gibson2\assets\dataset\processed\canned_food\10\rigid_body.urdf'
scale = 0.8162
object_pos = [1, -0.4, 1]
object_orn = [0.707, 0.707, 0, 0]
obj3 = InteractiveObj(filename=fname, scale=scale)
s.import_object(obj3)
obj3.set_position_orientation(object_pos, object_orn)

fname = r'C:\Users\igibs\iGibson\gibson2\assets\dataset\processed\canned_food\14\rigid_body.urdf'
scale = 0.7954
object_pos = [1, -0.6, 1]
object_orn = [0.707, 0.707, 0, 0]
obj4 = InteractiveObj(filename=fname, scale=scale)
s.import_object(obj4)
obj4.set_position_orientation(object_pos, object_orn)

fname = r'C:\Users\igibs\iGibson\gibson2\assets\dataset\processed\canned_food\18\rigid_body.urdf'
scale = 0.8194
object_pos = [1, -0.8, 1]
object_orn = [0.707, 0.707, 0, 0]
obj5 = InteractiveObj(filename=fname, scale=scale)
s.import_object(obj5)
obj5.set_position_orientation(object_pos, object_orn)

fname = r'C:\Users\igibs\iGibson\gibson2\assets\dataset\processed\dairy\24\rigid_body.urdf'
scale = 0.81
object_pos = [1, -1.0, 1]
object_orn = [0.44, 0.44, -0.55, -0.55]
obj6 = InteractiveObj(filename=fname, scale=scale)
s.import_object(obj6)
obj6.set_position_orientation(object_pos, object_orn)


if optimize:
    s.optimize_data()

# Start user close to counter for interaction
s.setVROffset([1.0, 0, -0.4])

while True:
    s.step(shouldTime=False)

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
