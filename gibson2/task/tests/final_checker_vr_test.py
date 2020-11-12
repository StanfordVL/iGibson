from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.utils.utils import parse_config
from gibson2.utils.vr_utils import move_player_no_body
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.task.task_base import iGTNTask
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from tasknet.object import BaseObject
import os
import gibson2
import time
import random
import sys
import numpy as np
import pybullet as p

optimize = True

# HDR files for PBR rendering
scene_name = 'Beechwood_0_int'

hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', scene_name, 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

scene = InteractiveIndoorScene(
    scene_name, texture_randomization=False, object_randomization=False)

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
# s.viewer.min_cam_z = 1.0


# List of object names to filename mapping
sim_objects = []
dsl_objects = []

lunch_pack_folder = os.path.join(gibson2.assets_path, 'dataset', 'processed', 'pack_lunch')
lunch_pack_files = {
    'chips': os.path.join(lunch_pack_folder, 'food', 'snack', 'chips', 'chips0', 'rigid_body.urdf'),
    'fruit': os.path.join(lunch_pack_folder, 'food', 'fruit', 'pear', 'pear00', 'rigid_body.urdf'),
    'soda': os.path.join(lunch_pack_folder, 'drink', 'soda', 'soda23_mountaindew710mL', 'rigid_body.urdf'),
    'eggs': os.path.join(lunch_pack_folder, 'food', 'protein', 'eggs', 'eggs00_eggland', 'rigid_body.urdf'),
    'container': os.path.join(lunch_pack_folder, 'dish', 'casserole_dish', 'casserole_dish00', 'rigid_body.urdf')
}

item_scales = {
    'chips': 1,
    'fruit': 0.9,
    'soda': 0.8,
    'eggs': 0.5,
    'container': 0.5
}

# A list of start positions and orientations for the objects - determined by placing objects in VR
item_start_pos_orn = {
    'chips': [
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
    'soda': [
        [(-5.0, -3.55, 1.03), (0.68, -0.18, -0.18, 0.68)],
        [(-5.0, -3.7, 1.03), (0.68, -0.18, -0.18, 0.68)],
        [(-5.0, -3.85, 1.03), (0.68, -0.18, -0.18, 0.68)],
        [(-5.0, -4.0, 1.03), (0.68, -0.18, -0.18, 0.68)],
    ],
    'eggs': [
        [(-4.65, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.66, -1.58, 1.46), (0.72, 0, 0, 0.71)],
        [(-4.89, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.89, -1.58, 1.46), (0.72, 0, 0, 0.71)],
    ],
    'container': [
        [(-4.1, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.5, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.9, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-5.3, -1.82, 0.87), (0.71, 0, 0, 0.71)],
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
        sim_objects.append(item_ob)
        dsl_objects.append(BaseObject(item))
        if item == 'container':
            p.changeDynamics(item_ob.body_id, -1, mass=8., lateralFriction=0.9)

vr_body = VrBody()
s.import_object(vr_body, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
vr_body.init_body([kitchen_middle[0], kitchen_middle[1]])

r_hand = VrHand(hand='right')
s.import_object(r_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
r_hand.set_start_state(start_pos=[kitchen_middle[0], kitchen_middle[1], 2])

l_hand = VrHand(hand='left')
s.import_object(l_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
l_hand.set_start_state(start_pos=[kitchen_middle[0], kitchen_middle[1], 2.2])

# Set up task 
igtn_task = iGTNTask('pack_lunch_demo', task_instance=2)
igtn_task.initialize_simulator(handmade_simulator=s,
                           handmade_sim_objs=sim_objects,
                           handmade_dsl_objs=dsl_objects)

# VR loop
time_fps = True
if optimize:
    s.optimize_vertex_and_texture()

s.set_vr_offset([-4.34, -2.68, -0.5])

while True:
    # Step simulator 
    start_time = time.time()
    igtn_task.simulator.step()

    # Handle VR elements
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



    # Check success 
    success, failed_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', failed_conditions)
    else:
        # break
        pass
    print('\n')

    frame_dur = time.time() - start_time
    if time_fps:
        print('Fps: {}'.format(round(1/max(frame_dur, 0.00001), 2)))
    
igtn_task.simulator.disconnect()


