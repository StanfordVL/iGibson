""" A very simple VR program containing only a single scene.
The user can fly around the scene using the controller, and can
explore whether all the graphics features of iGibson are working as intended.
"""

import numpy as np
import os
import pybullet as p
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import move_player_no_body

optimize = True

# HDR files for PBR rendering
hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Beechwood_0_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

# VR rendering settings
vr_rendering_settings = MeshRendererSettings(optimized=optimize,
                                            fullscreen=False,
                                            env_texture_filename=hdr_texture,
                                            env_texture_filename2=hdr_texture2,
                                            env_texture_filename3=background_texture,
                                            light_modulation_map_filename=light_modulation_map_filename,
                                            enable_shadow=False, 
                                            enable_pbr=True,
                                            msaa=True,
                                            light_dimming_factor=1.0)
# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, rendering_settings=vr_rendering_settings,
            vr_eye_tracking=False, vr_mode=True)
scene = InteractiveIndoorScene('Beechwood_0_int')
scene._set_first_n_objects(5)
s.import_ig_scene(scene)

# This is used to place the objects easily
p.setGravity(0, 0, 0)

# Position that is roughly in the middle of the kitchen - used to help place objects
kitchen_middle = [-4.5, -3.5, 1.5]

# List of object names to filename mapping
lunch_pack_folder = os.path.join(gibson2.assets_path, 'pack_lunch')
lunch_pack_files = {
    'sandwich': os.path.join(lunch_pack_folder, 'cereal', 'cereal01', 'rigid_body.urdf'),
    'chip': os.path.join(lunch_pack_folder, 'food', 'snack', 'chips', 'chips0', 'rigid_body.urdf'),
    'fruit': os.path.join(lunch_pack_folder, 'food', 'fruit', 'pear', 'pear00', 'rigid_body.urdf'),
    'bread': os.path.join(lunch_pack_folder, 'granola', 'granola00', 'rigid_body.urdf'),
    'yogurt': os.path.join(lunch_pack_folder, 'food', 'dairy', 'yogurt', 'yogurt00_dannonbananacarton', 'rigid_body.urdf'),
    'water': os.path.join(lunch_pack_folder, 'drink', 'soda', 'soda23_mountaindew710mL', 'rigid_body.urdf'),
    'eggs': os.path.join(lunch_pack_folder, 'eggs', 'eggs00_eggland', 'rigid_body.urdf'),
    'container': os.path.join(lunch_pack_folder, 'dish', 'casserole_dish', 'casserole_dish00', 'rigid_body.urdf')
}

item_scales = {
    'sandwich': 1,
    'chip': 1,
    'fruit': 1,
    'bread': 1,
    'yogurt': 1,
    'water': 0.8,
    'eggs': 0.5,
    'container': 0.4
}

# A list of start positions and orientations for the objects - determined by placing objects in VR
item_start_pos_orn = {
    'sandwich': [
        [(-4.55, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.65, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.75, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.85, -1.65, 1.5), (0, 1, 1, 1)],
    ],
    'chip': [
        [(-4.55, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.65, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.75, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.85, -1.65, 1.5), (0, 1, 1, 1)],
    ],
    'fruit': [
        [(-4.55, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.65, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.75, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.85, -1.65, 1.5), (0, 1, 1, 1)],
    ],
    'bread': [
        [(-4.55, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.65, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.75, -1.65, 1.5), (0, 1, 1, 1)],
        [(-4.85, -1.65, 1.5), (0, 1, 1, 1)],
    ],
    'yogurt': [
        [(-5.43, -1.64, 1.68), (0.57, 0.42, 0.42, 0.57)],
        [(-5.32, -1.64, 1.68), (0.57, 0.42, 0.42, 0.57)],
        [(-5.2, -1.64, 1.68), (0.57, 0.42, 0.42, 0.57)],
        [(-5.1, -1.64, 1.68), (0.57, 0.42, 0.42, 0.57)],
    ],
    'water': [
        [(-4.61, -1.69, 1.73), (0.68, -0.18, -0.18, 0.68)],
        [(-4.69, -1.69, 1.73), (0.68, -0.18, -0.18, 0.68)],
        [(-4.8, -1.69, 1.73), (0.68, -0.18, -0.18, 0.68)],
        [(-4.9, -1.69, 1.73), (0.68, -0.18, -0.18, 0.68)],
    ],
    'eggs': [
        [(-4.65, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.66, -1.58, 1.46), (0.72, 0, 0, 0.71)],
        [(-4.96, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.96, -1.58, 1.46), (0.72, 0, 0, 0.71)],
    ],
    'container': [
        [(-4.16, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.53, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.9, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-5.3, -1.82, 0.87), (0.71, 0, 0, 0.71)],
    ]
}

finished_items = ['yogurt', 'water', 'eggs', 'container']

# Import all objects and put them in the correct positions
pack_items = list(lunch_pack_files.keys())
for item in pack_items:
    # Just load in sandwiches for now
    if item not in finished_items:
        continue
    fpath = lunch_pack_files[item]
    start_pos_orn = item_start_pos_orn[item]
    item_scale = item_scales[item]
    for pos, orn in start_pos_orn:
        item_ob = ArticulatedObject(fpath, scale=item_scale)
        s.import_object(item_ob, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

r_hand = VrHand(hand='right')
s.import_object(r_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
r_hand.set_start_state(start_pos=[kitchen_middle[0], kitchen_middle[1], 2])

l_hand = VrHand(hand='left')
s.import_object(l_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
l_hand.set_start_state(start_pos=[kitchen_middle[0], kitchen_middle[1], 2.2])

if optimize:
    s.optimize_vertex_and_texture()

s.set_vr_offset([-4.34, -2.68, 0.0])

time_fps = False
while True:
    # Use right menu controller to change z offset to we can easily change height for placing objects
    vr_z_offset = 0
    event_list = s.poll_vr_events()
    for event in event_list:
        device_type, event_type = event
        if device_type == 'right_controller':
            if event_type == 'menu_press':
                # Press the menu button to move up
                vr_z_offset = 0.01
            elif event_type == 'grip_press':
                # Press the grip to move down
                vr_z_offset = -0.01

    curr_offset = s.get_vr_offset()
    s.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])

    start_time = time.time()
    s.step()

    hmd_is_valid, hmd_trans, hmd_rot = s.get_data_for_vr_device('hmd')
    l_is_valid, l_trans, l_rot = s.get_data_for_vr_device('left_controller')
    r_is_valid, r_trans, r_rot = s.get_data_for_vr_device('right_controller')
    l_trig, l_touch_x, l_touch_y = s.get_button_data_for_controller('left_controller')
    r_trig, r_touch_x, r_touch_y = s.get_button_data_for_controller('right_controller')
    if r_is_valid:
        r_hand.move(r_trans, r_rot)
        r_hand.set_close_fraction(r_trig)
        move_player_no_body(s, r_touch_x, r_touch_y, 0.01, 'hmd')
    
    if l_is_valid:
        l_hand.move(l_trans, l_rot)
        l_hand.set_close_fraction(l_trig)

    frame_dur = time.time() - start_time
    if time_fps:
        print('Fps: {}'.format(round(1/max(frame_dur, 0.00001), 2)))

s.disconnect()