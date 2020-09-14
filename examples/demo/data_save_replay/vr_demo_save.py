# Stores PyBullet and VR data using the hdf5 file format
import pybullet as p
import numpy as np

from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import InteractiveObj, VrHand, YCBObject
from gibson2.core.simulator import Simulator
from gibson2 import assets_path
from gibson2.utils.vr_logging import VRLogWriter

model_path = assets_path + '\\models'

optimize = True

s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

rHand = VrHand(start_pos=[0.0, 0.5, 1.5], leftHand=False)
s.import_articulated_object(rHand)

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

s.setVROffset([1.0, 0, -0.4])

# The VRLogWriter has a simple interface:
# all we have to do is initialize it
# and then call process_frame at the end of
# each frame to record data.
# Data is automatically flushed every
# frames_before_write frames to hdf5.
vr_log_path = 'vr_logs/vr_demo_save.h5'
# Saves every 2 seconds or so
vr_writer = VRLogWriter(frames_before_write=200, log_filepath=vr_log_path, profiling_mode=True)

# 2000 frames corresponds to approximately 20-30 seconds of data collection
for i in range(2000):
    s.step(shouldPrintTime=False)

    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')
    rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

    if rIsValid:
        rHand.move_hand(rTrans, rRot)
        rHand.toggle_finger_state(rTrig)

    vr_writer.process_frame(s)

# Note: always call this after the simulation is over to close the log file
# and clean up resources used.
vr_writer.end_log_session()

s.disconnect()