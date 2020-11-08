""" VR replay demo using simplified VR playground code.

This demo replay the states of all objects in their entirety, and does
not involve any meaningful physical simulation.

Note: This demo does not use PBR so it can be supported on a wide range of devices, including Mac OS.

This demo reads logs from to vr_logs/vr_demo_save_states.h5
If you would like to replay your own data, please run
vr_demo_save_states and change the file path where data is recoded."""

import numpy as np
import os
import pybullet as p

from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_logging import VRLogReader
from gibson2.utils.vr_utils import move_player_no_body
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# Playground configuration: edit this to change functionality
optimize = True
# Toggles fullscreen companion window
fullscreen = False
# Toggles SRAnipal eye tracking
use_eye_tracking = True
# Enables the VR collision body
enable_vr_body = True
# Toggles movement with the touchpad (to move outside of play area)
touchpad_movement = True
# Set to one of hmd, right_controller or left_controller to move relative to that device
relative_movement_device = 'hmd'
# Movement speed for touchpad-based movement
movement_speed = 0.03

# Initialize simulator with specific rendering settings
s = Simulator(mode='simple', image_width=504, image_height=560,
            rendering_settings=MeshRendererSettings(optimized=optimize, fullscreen=fullscreen, enable_pbr=False))
#s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, 
#            rendering_settings=MeshRendererSettings(optimized=optimize, fullscreen=fullscreen, enable_pbr=False),
#            vr_eye_tracking=use_eye_tracking, vr_mode=False)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

# Player body is represented by a translucent blue cylinder
if enable_vr_body:
    vr_body = VrBody()
    s.import_object(vr_body)
    # Note: we don't call init_body for the VR body to avoid constraints interfering with the replay

# The hand can either be 'right' or 'left'
# It has enough friction to pick up the basket and the mustard bottles
r_hand = VrHand(hand='right')
s.import_object(r_hand)
# Note: we don't call set start state for the VR hands to avoid constraints interfering with the replay

l_hand = VrHand(hand='left')
s.import_object(l_hand)
# Note: we don't call set start state for the VR hands to avoid constraints interfering with the replay

if use_eye_tracking:
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
mustard_list = []
for i in range(len(mass_list)):
    mustard = YCBObject('006_mustard_bottle')
    mustard_list.append(mustard)
    s.import_object(mustard)
    mustard.set_position([mustard_start[0], mustard_start[1] - i * 0.2, mustard_start[2]])
    p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

if optimize:
    s.optimize_vertex_and_texture()

# Note: the VRLogReader plays back the demo at the recorded fps, so there is not need to set this
vr_log_path = 'vr_logs/vr_demo_save_states.h5'
vr_reader = VRLogReader(log_filepath=vr_log_path)

# The VR reader automatically shuts itself down and performs cleanup once the while loop has finished running
while vr_reader.get_data_left_to_read():
    # We need to read frame before step for various reasons - one of them is that we need to set the camera
    # matrix for this frame before rendering in step
    vr_reader.read_frame(s, fullReplay=True)
    s.step()