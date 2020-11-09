""" Lunch packing demo - initial conditions - Eric """

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
                                            enable_shadow=True, 
                                            enable_pbr=True,
                                            msaa=True,
                                            light_dimming_factor=1.0)
# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', rendering_settings=vr_rendering_settings, vr_eye_tracking=False, vr_mode=True)
scene = InteractiveIndoorScene('Beechwood_0_int')
s.import_ig_scene(scene)

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
    'sandwich': 0.7,
    'chip': 1,
    'fruit': 0.9,
    'bread': 0.7,
    'yogurt': 1,
    'water': 0.8,
    'eggs': 0.5,
    'container': 0.3
}

# A list of start positions and orientations for the objects - determined by placing objects in VR
item_start_pos_orn = {
    'sandwich': [
        [(-5.24, -1.6, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.24, -1.7, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.24, -1.8, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.24, -1.9, 0.97), (0, 0.71, 0.71, 0)],
    ],
    'chip': [
        [(-5.39, -1.62, 1.42), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.39, -1.62, 1.49), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.12, -1.62, 1.42), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.12, -1.62, 1.49), (-0.14, -0.06, 0.71, 0.69)],
    ],
    'fruit': [
        [(-4.8, -3.55, 0.97), (0, 0, 0, 1)],
        [(-4.8, -3.7, 0.97), (0, 0, 0, 1)],
        [(-4.8, -3.85, 0.97), (0, 0, 0, 1)],
        [(-4.8, -4.0, 0.97), (0, 0, 0, 1)],
    ],
    'bread': [
        [(-5.39, -1.6, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.39, -1.7, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.39, -1.8, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.39, -1.9, 0.97), (0, 0.71, 0.71, 0)],
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
        [(-4.89, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.89, -1.58, 1.46), (0.72, 0, 0, 0.71)],
    ],
    'container': [
        [(-4.1, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.4, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.7, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-5.0, -1.82, 0.87), (0.71, 0, 0, 0.71)],
    ]
}

# Import all objects and put them in the correct positions
pack_items = list(lunch_pack_files.keys())
for item in pack_items:
    fpath = lunch_pack_files[item]
    start_pos_orn = item_start_pos_orn[item]
    item_scale = item_scales[item]
    for pos, orn in start_pos_orn:
        item_ob = ArticulatedObject(fpath, scale=item_scale)
        s.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

vr_body = VrBody()
s.import_object(vr_body, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
vr_body.init_body([kitchen_middle[0], kitchen_middle[1]])

r_hand = VrHand(hand='right')
s.import_object(r_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
r_hand.set_start_state(start_pos=[kitchen_middle[0], kitchen_middle[1], 2])

l_hand = VrHand(hand='left')
s.import_object(l_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
l_hand.set_start_state(start_pos=[kitchen_middle[0], kitchen_middle[1], 2.2])

if optimize:
    s.optimize_vertex_and_texture()

s.set_vr_offset([-4.34, -2.68, -0.5])

time_fps = True
while True:
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
        vr_body.move_body(s, r_touch_x, r_touch_y, 0.03, 'hmd')
    
    if l_is_valid:
        l_hand.move(l_trans, l_rot)
        l_hand.set_close_fraction(l_trig)

    frame_dur = time.time() - start_time
    if time_fps:
        print('Fps: {}'.format(round(1/max(frame_dur, 0.00001), 2)))

s.disconnect()