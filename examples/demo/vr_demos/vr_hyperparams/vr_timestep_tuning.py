""" VR playground containing various objects and VR options that can be toggled
to experiment with the VR experience in iGibson. This playground operates in a
PBR scene. Please see vr_playground_no_pbr.py for a non-PBR experience.

Important - VR functionality and where to find it:

1) Most VR functions can be found in the gibson2/simulator.py
2) VR utility functions are found in gibson2/utils/vr_utils.py
3) The VR renderer can be found in gibson2/render/mesh_renderer.py
4) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in gibson2/render/cpp
"""

import numpy as np
import os
import pybullet as p
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.object_base import Object
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import move_player_no_body
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')
groceries_folder = os.path.join(assets_path, 'models', 'groceries')

# Playground configuration: edit this to change functionality
optimize = True
# Toggles fullscreen companion window
fullscreen = False
# Toggles SRAnipal eye tracking
use_eye_tracking = True
# Toggles movement with the touchpad (to move outside of play area)
touchpad_movement = True
# Set to one of hmd, right_controller or left_controller to move relative to that device
relative_movement_device = 'hmd'
# Movement speed for touchpad-based movement
movement_speed = 0.03

# HDR files for PBR rendering
hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

# VR rendering settings
vr_rendering_settings = MeshRendererSettings(optimized=optimize,
                                            fullscreen=fullscreen,
                                            env_texture_filename=hdr_texture,
                                            env_texture_filename2=hdr_texture2,
                                            env_texture_filename3=background_texture,
                                            light_modulation_map_filename=light_modulation_map_filename,
                                            enable_shadow=True, 
                                            enable_pbr=True,
                                            msaa=True,
                                            light_dimming_factor=1.0)
# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', rendering_settings=vr_rendering_settings,
            vr_eye_tracking=use_eye_tracking, vr_mode=True)
scene = InteractiveIndoorScene('Rs_int')
# Turn this on when debugging to speed up loading
scene._set_first_n_objects(10)
s.import_ig_scene(scene)

# The hand can either be 'right' or 'left'
# It has enough friction to pick up the basket and the mustard bottles
r_hand = VrHand(hand='right')
s.import_object(r_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
# This sets the hand constraints so it can move with the VR controller
r_hand.set_start_state(start_pos=[0, 0, 1.5])

l_hand = VrHand(hand='left')
s.import_object(l_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
# This sets the hand constraints so it can move with the VR controller
l_hand.set_start_state(start_pos=[0, 0.5, 1.5])

if use_eye_tracking:
    # Eye tracking visual marker - a red marker appears in the scene to indicate gaze direction
    gaze_marker = VisualMarker(radius=0.03)
    s.import_object(gaze_marker, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
    gaze_marker.set_position([0,0,1.5])

basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path, scale=0.8)
s.import_object(basket)
basket.set_position([-1, 1.55, 1.2])
p.changeDynamics(basket.body_id, -1, mass=5)

can_1_path = os.path.join(groceries_folder, 'canned_food', '1', 'rigid_body.urdf')
can_pos = [[-0.8, 1.55, 1.2], [-0.6, 1.55, 1.2], [-0.4, 1.55, 1.2]]
cans = []
for i in range (len(can_pos)):
    can_1 = ArticulatedObject(can_1_path, scale=0.6)
    cans.append(can_1)
    s.import_object(can_1)
    can_1.set_position(can_pos[i])

if optimize:
    s.optimize_vertex_and_texture()

# Set VR starting position in the scene
s.set_vr_offset([0, 0, -0.1])

while True:
    s.step(print_time=True)

    # VR device data
    hmd_is_valid, hmd_trans, hmd_rot = s.get_data_for_vr_device('hmd')
    l_is_valid, l_trans, l_rot = s.get_data_for_vr_device('left_controller')
    r_is_valid, r_trans, r_rot = s.get_data_for_vr_device('right_controller')

    # VR button data
    l_trig, l_touch_x, l_touch_y = s.get_button_data_for_controller('left_controller')
    r_trig, r_touch_x, r_touch_y = s.get_button_data_for_controller('right_controller')

    # VR eye tracking data
    if use_eye_tracking:
        is_eye_data_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.get_eye_tracking_data()
        if is_eye_data_valid:
            # Move gaze marker based on eye tracking data
            updated_marker_pos = [origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]]
            gaze_marker.set_position(updated_marker_pos)

    if r_is_valid:
        r_hand.move(r_trans, r_rot)
        r_hand.set_close_fraction(r_trig)

        # Right hand used to control movement
        # Move VR system based on device coordinate system and touchpad press location
        move_player_no_body(s, r_touch_x, r_touch_y, movement_speed, relative_movement_device)

        # Trigger haptic pulse on right touchpad, modulated by trigger close fraction
        # Close the trigger to create a stronger pulse
        # Note: open trigger has closed fraction of 0.05 when open, so cutoff haptic input under 0.1
        # to avoid constant rumbling
        s.trigger_haptic_pulse('right_controller', r_trig if r_trig > 0.1 else 0)

    if l_is_valid:
        l_hand.move(l_trans, l_rot)
        l_hand.set_close_fraction(l_trig)
        s.trigger_haptic_pulse('left_controller', l_trig if l_trig > 0.1 else 0)

s.disconnect()