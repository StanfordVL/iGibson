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
optimize = False
# Toggles fullscreen companion window
fullscreen = False

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
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, rendering_settings=vr_rendering_settings,
            vr_eye_tracking=False, vr_mode=True)
scene = InteractiveIndoorScene('Rs_int')
# Turn this on when debugging to speed up loading
scene._set_first_n_objects(5)
s.import_ig_scene(scene)

# TODO: Remove later
p.setGravity(0, 0, 0)

# Player body is represented by a translucent blue cylinder
""" if enable_vr_body:
    vr_body = VrBody()
    s.import_object(vr_body, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
    vr_body.init_body([0,0]) """

vr_body_fpath = os.path.join(assets_path, 'models', 'vr_body', 'vr_body.urdf')
vrb = ArticulatedObject(vr_body_fpath, scale=0.1)
s.import_object(vrb, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
#vrb.set_position([0, 0, 1])
vrb_cid = p.createConstraint(vrb.body_id, -1, -1, -1, p.JOINT_FIXED, 
                                            [0, 0, 0], [0, 0, 0], [0, 0, 1.2])

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

# TODO: Remove this test
#r_hand.set_hand_no_collision(can_1.body_id)
#r_hand.set_hand_no_collision(basket.body_id)
#r_hand.set_hand_no_collision(vr_body.body_id)
#p.setCollisionFilterPair(can_1.body_id, basket.body_id, -1, -1, 0) # the last argument is 0 for disabling collision, 1 for enabling collision
#p.setCollisionFilterPair(can_1.body_id, r_hand.body_id, -1, -1, 0)
#p.setCollisionFilterPair(can_1.body_id, l_hand.body_id, -1, -1, 0)

if optimize:
    s.optimize_vertex_and_texture()

# Set VR starting position in the scene
s.set_vr_offset([0, 0, 0])

while True:
    s.step()
    hmd_is_valid, hmd_trans, hmd_rot = s.get_data_for_vr_device('hmd')
    l_is_valid, l_trans, l_rot = s.get_data_for_vr_device('left_controller')
    r_is_valid, r_trans, r_rot = s.get_data_for_vr_device('right_controller')
    l_trig, l_touch_x, l_touch_y = s.get_button_data_for_controller('left_controller')
    r_trig, r_touch_x, r_touch_y = s.get_button_data_for_controller('right_controller')

    if hmd_is_valid:
        p.changeConstraint(vrb_cid, hmd_trans, vrb.get_orientation(), maxForce=2000)


    """if enable_vr_body:
        if not r_is_valid:
            # See VrBody class for more details on this method
            vr_body.move_body(s, 0, 0, movement_speed, relative_movement_device)
        else:
            vr_body.move_body(s, r_touch_x, r_touch_y, movement_speed, relative_movement_device) """

    if r_is_valid:
        r_hand.move(r_trans, r_rot)
        r_hand.set_close_fraction(r_trig)

        # Right hand used to control movement
        # Move VR system based on device coordinate system and touchpad press location
        move_player_no_body(s, r_touch_x, r_touch_y, 0.03, 'hmd')

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