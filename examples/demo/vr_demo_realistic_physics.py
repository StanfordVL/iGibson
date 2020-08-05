import pybullet as p
import numpy as np
import time

from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import YCBObject, InteractiveObj, VrHand
from gibson2.core.simulator import Simulator
from gibson2 import assets_path
from gibson2.utils.vr_utils import translate_vr_position_by_vecs

model_path = assets_path + '\\models'
sample_urdf_folder = model_path + '\\sample_urdfs\\'

optimize = True

s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

# VR hand object
rHand = VrHand(start_pos=[0.0, 0.5, 1.5])
s.import_articulated_object(rHand)

# Heavy robot
heavy_robot = InteractiveObj(sample_urdf_folder + 'object_H3ygj6efM8V.urdf')
s.import_object(heavy_robot)
heavy_robot.set_position([1, 0.2, 1])
# Robot is just a base, so use -1 as link index for changing dynamics
p.changeDynamics(heavy_robot.body_id, -1, mass=500)

# Medium robot
medium_robot = InteractiveObj(sample_urdf_folder + 'object_H3ygj6efM8V.urdf')
s.import_object(medium_robot)
medium_robot.set_position([1, 0, 1])
# Robot is just a base, so use -1 as link index for changing dynamics
p.changeDynamics(medium_robot.body_id, -1, mass=50)

# Light robot
light_robot = InteractiveObj(sample_urdf_folder + 'object_H3ygj6efM8V.urdf')
s.import_object(light_robot)
light_robot.set_position([1, -0.2, 1])

# Heavy mustard
heavy_bottle = YCBObject('006_mustard_bottle')
s.import_object(heavy_bottle)
heavy_bottle.set_position([1, -0.4, 1])
p.changeDynamics(heavy_bottle.body_id, -1, mass=500)

# Light mustard
light_bottle = YCBObject('006_mustard_bottle')
s.import_object(light_bottle)
light_bottle.set_position([1, -0.6, 1])

if optimize:
    s.optimize_data()

# Start user close to counter for interaction
s.setVROffset([1.0, 0, -0.4])

while True:
    s.step(shouldTime=False)

    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')
    rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

    relative_device = 'hmd'
    right, _, forward = s.getDeviceCoordinateSystem(relative_device)

    if rIsValid:
        rHand.move_hand(rTrans, rRot)
        rHand.toggle_finger_state(rTrig)

        new_offset = translate_vr_position_by_vecs(rTouchX, rTouchY, right, forward, s.getVROffset(), 0.01)
        s.setVROffset(new_offset)

s.disconnect()