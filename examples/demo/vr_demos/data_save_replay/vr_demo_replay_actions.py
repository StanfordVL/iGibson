""" VR replay demo using simplified VR playground code.

This demo runs the log saved at vr_logs/vr_demo_save.h5"""

import numpy as np
import os
import pybullet as p

from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_logging import VRLogReader
from gibson2.utils.vr_utils import translate_vr_position_by_vecs
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# Playground configuration: edit this to change functionality
optimize = True
# Toggles fullscreen companion window
fullscreen = False

# Initialize simulator
s = Simulator(mode='vr', timestep = 1/90.0, optimized_renderer=optimize, vrFullscreen=fullscreen, vrMode=False)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

rHand = VrHand()
s.import_object(rHand)
# Note: We do not call set_start_state, as this would add in a constraint that messes up data replay

# Add playground objects to the scene
# Eye tracking visual marker - a red marker appears in the scene to indicate gaze direction
gaze_marker = VisualMarker(radius=0.03)
s.import_object(gaze_marker)
gaze_marker.set_position([0,0,1.5])

basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path)
s.import_object(basket)
basket.set_position([1, 0.2, 1])
p.changeDynamics(basket.body_id, -1, mass=5)

mass_list = [5, 10, 100, 500]
mustard_start = [1, -0.2, 1]
for i in range(len(mass_list)):
    mustard = YCBObject('006_mustard_bottle')
    s.import_object(mustard)
    mustard.set_position([mustard_start[0], mustard_start[1] - i * 0.2, mustard_start[2]])
    p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

if optimize:
    s.optimize_vertex_and_texture()

# Note: the VRLogReader plays back the demo at the recorded fps, so there is not need to set this
vr_log_path = 'vr_logs/vr_demo_save.h5'
vr_reader = VRLogReader(log_filepath=vr_log_path)
vr_hand_action_path = 'vr_hand'

while vr_reader.get_data_left_to_read():
    vr_hand_actions = vr_reader.read_action(vr_hand_action_path)
    print(vr_hand_actions.shape, vr_hand_actions)

    vr_reader.read_frame(s, fullReplay=False)
    s.step(shouldPrintTime=False)