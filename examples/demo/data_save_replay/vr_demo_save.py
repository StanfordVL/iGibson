import pybullet as p
import numpy as np
import h5py

from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import InteractiveObj, VrHand
from gibson2.core.simulator import Simulator
from gibson2 import assets_path
from vr_logger import VRLogWriter

model_path = assets_path + '\\models'
sample_urdf_folder = model_path + '\\sample_urdfs\\'

optimize = True

s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

rHand = VrHand(start_pos=[0.0, 0.5, 1.5])
s.import_articulated_object(rHand)

model = InteractiveObj(sample_urdf_folder + 'object_H3ygj6efM8V.urdf')
s.import_object(model)
model.set_position([1,0,2])

if optimize:
    s.optimize_data()

s.setVROffset([1.0, 0, -0.4])

# HDF5 test:
d1 = np.random.random(size = (1000,20))
hf = h5py.File('data/data.h5', 'w')
hf.create_dataset('dataset_1', data=d1)

# Write file to disk
hf.close()

hf = h5py.File('data/data.h5', 'r')
n1 = hf.get('dataset_1')
n1 = np.array(n1)
print(n1)
hf.close()

# The VRLogWriter has a simple interface:
# all we have to do is initialize it
# and then call process_frame at the end of
# each frame to record data.
# Data is automatically flushed every
# frames_before_write frames to hdf5.
vr_writer = VRLogWriter(frames_before_write=5)

# Small number of frames to test log writer class!
for i in range(8):
    s.step(shouldPrintTime=False)

    hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')

    lTrig, lTouchX, lTouchY = s.getButtonDataForController('left_controller')
    rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

    if rIsValid:
        rHand.move_hand(rTrans, rRot)
        rHand.toggle_finger_state(rTrig)

    vr_writer.process_frame(s)

s.disconnect()