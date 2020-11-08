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
#scene._set_first_n_objects(10)
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

item_nums = {
    'sandwich': 4,
    'chip': 4,
    'fruit': 4,
    'bread': 4,
    'yogurt': 4,
    'water': 4,
    'eggs': 4,
    'container': 4
}

# Objects will be loaded into a grid starting at the kitchen middle
# All have the same starting x coordinate and different y coordinates (offsets from kitchen middle)
item_y_offsets = {
    'sandwich': 0.4,
    'chip': 0.1,
    'fruit': -0.4,
    'bread': -0.6,
    'yogurt': -0.8,
    'water': -1.0,
    'eggs': -1.4,
    'container': -1.8
}
x_offsets = [0, -0.2, -0.4, -0.6]
item_height = 1.2

# Store object data for body id - name, position and orientation - for use in object placement
body_id_dict = {}

all_items = []
# Import all objects and put them in the correct positions
pack_items = list(lunch_pack_files.keys())
for item in pack_items:
    fpath = lunch_pack_files[item]
    y_offset = item_y_offsets[item]
    num_items = item_nums[item]
    for i in range(num_items):
        x_offset = x_offsets[i]
        if item == 'container':
            x_offset = x_offset * 2
        if item == 'eggs':
            x_offset = x_offset * 1.7
        item_scale = 1
        if item == 'container':
            item_scale = 0.4
        elif item == 'eggs':
            item_scale = 0.5
        elif item == 'water':
            item_scale = 0.8
        item_ob = ArticulatedObject(fpath, scale=item_scale)
        s.import_object(item_ob, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
        item_ob.set_position([kitchen_middle[0] + x_offset, kitchen_middle[1] + y_offset, item_height])
        all_items.append(item_ob)
        bid = item_ob.body_id
        item_data = [item, item_ob.get_position(), item_ob.get_orientation()]
        body_id_dict[bid] = item_data

r_hand = VrHand(hand='right')
s.import_object(r_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
r_hand.set_start_state(start_pos=[kitchen_middle[0], kitchen_middle[1], 2])

l_hand = VrHand(hand='left')
s.import_object(l_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
l_hand.set_start_state(start_pos=[kitchen_middle[0], kitchen_middle[1], 2.2])

if optimize:
    s.optimize_vertex_and_texture()

s.set_vr_pos([-4.5, -3.5, 0.3])

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
            elif event_type == 'touchpad_press':
                # Print body_id_dict data
                print(body_id_dict)
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
        move_player_no_body(s, r_touch_x, r_touch_y, 0.005, 'hmd')
    
    if l_is_valid:
        l_hand.move(l_trans, l_rot)
        l_hand.set_close_fraction(l_trig)

    frame_dur = time.time() - start_time
    if time_fps:
        print('Fps: {}'.format(round(1/max(frame_dur, 0.00001), 2)))

    # Every frame we update the body_id_dictionary data
    for item in all_items:
        body_id_dict[item.body_id][1] = item.get_position()
        body_id_dict[item.body_id][2] = item.get_orientation()

s.disconnect()