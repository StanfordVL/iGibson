# Replays the data stored in vr_demo_save.py
import pybullet as p
import numpy as np

from gibson2.core.physics.scene import BuildingScene
from gibson2.core.physics.interactive_objects import InteractiveObj, VrHand, YCBObject
from gibson2.core.simulator import Simulator
from gibson2 import assets_path
from gibson2.utils.vr_logging import VRLogReader

model_path = assets_path + '\\models'
sample_urdf_folder = model_path + '\\sample_urdfs\\'

optimize = True

# Note: the set-up code is all the same as in vr_demo_save
# Make sure to set VR mode to false when doing data replay!
s = Simulator(mode='vr', timestep = 1/90.0, msaa=True, vrFullscreen=False, vrEyeTracking=False, optimize_render=optimize, vrMode=False)
scene = BuildingScene('Placida', is_interactive=False)
scene.sleep = optimize
s.import_scene(scene)

rHand = VrHand(start_pos=[0.0, 0.5, 1.5], leftHand=False, replayMode=True)
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

# TODO: Remove this later!
camera_pose = np.array([0, 0, 1.2])
view_direction = np.array([1, 0, 0])
s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
s.renderer.set_fov(90)

vr_log_path = 'vr_logs/vr_demo_save.h5'
vr_reader = VRLogReader(log_filepath=vr_log_path, playback_fps=90)

while vr_reader.get_data_left_to_read():
    vr_reader.read_frame(s)

    # Call step to render the frame data that has just been read
    s.step(shouldPrintTime=False)
