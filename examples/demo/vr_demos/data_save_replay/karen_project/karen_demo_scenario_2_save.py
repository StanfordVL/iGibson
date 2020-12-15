""" Demo for Karen - scenario 2 - SAVING:

Instructions:
1) Walk from A to B
2) Pick up pepper bottle at B
3) Go to C and release pepper bottle

A = far end of the room
B = kitchen bar with chairs
C = stove top with another object
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
from gibson2.utils.vr_logging import VRLogWriter
from gibson2.utils.vr_utils import move_player_no_body
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')
groceries_folder = os.path.join(assets_path, 'models', 'groceries')

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
vr_rendering_settings = MeshRendererSettings(optimized=True,
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
s = Simulator(mode='vr', rendering_settings=vr_rendering_settings,
            vr_eye_tracking=True, vr_mode=True)
scene = InteractiveIndoorScene('Rs_int')
s.import_ig_scene(scene)

r_hand = VrHand(hand='right')
s.import_object(r_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
r_hand.set_start_state(start_pos=[0, 0, 1.5])

l_hand = VrHand(hand='left')
s.import_object(l_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
l_hand.set_start_state(start_pos=[0, 0.5, 1.5])

gaze_marker = VisualMarker(radius=0.03)
s.import_object(gaze_marker, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
gaze_marker.set_position([0,0,1.5])

basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path, scale=0.8)
s.import_object(basket)
basket.set_position([-1.8278704545622642, 2.152284546319316, 1.031713969848457])
p.changeDynamics(basket.body_id, -1, mass=5)

can_1_path = os.path.join(groceries_folder, 'canned_food', '1', 'rigid_body.urdf')
can_pos = [-0.8, 1.55, 1.1]
can_1 = ArticulatedObject(can_1_path, scale=0.6)
s.import_object(can_1)
can_1.set_position(can_pos)

s.optimize_vertex_and_texture()
# Set VR starting position in the scene
s.set_vr_offset([0, 0, -0.1])

# Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the third entry of the start pos
s.set_vr_start_pos([1.56085938, -2.88829452, 0], vr_height_offset=-0.1)

# Note: I appended "trial_n" manually to each log file, corresponding to the 3 trials I performed in each scenario
vr_log_path = 'data_logs/karen_demo_scenario_2.h5'
# Saves every 2 seconds or so (200 / 90fps is approx 2 seconds)
vr_writer = VRLogWriter(frames_before_write=200, log_filepath=vr_log_path, profiling_mode=False)

# Call set_up_data_storage once all actions have been registered (in this demo we only save states so there are none)
# Despite having no actions, we need to call this function
vr_writer.set_up_data_storage()

should_simulate = True
while should_simulate:
    event_list = s.poll_vr_events()
    for event in event_list:
        device_type, event_type = event
        if device_type == 'right_controller' or device_type == 'left_controller':
            if event_type == 'menu_press':
                # Quit data saving once the menu button has been pressed on either controller
                should_simulate = False
                
    s.step(print_time=False)

    # VR device data
    hmd_is_valid, hmd_trans, hmd_rot = s.get_data_for_vr_device('hmd')
    l_is_valid, l_trans, l_rot = s.get_data_for_vr_device('left_controller')
    r_is_valid, r_trans, r_rot = s.get_data_for_vr_device('right_controller')

    # VR button data
    l_trig, l_touch_x, l_touch_y = s.get_button_data_for_controller('left_controller')
    r_trig, r_touch_x, r_touch_y = s.get_button_data_for_controller('right_controller')

    is_eye_data_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.get_eye_tracking_data()
    if is_eye_data_valid:
        # Move gaze marker based on eye tracking data
        updated_marker_pos = [origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]]
        gaze_marker.set_position(updated_marker_pos)

    if r_is_valid:
        r_hand.move(r_trans, r_rot)
        r_hand.set_close_fraction(r_trig)
        move_player_no_body(s, r_touch_x, r_touch_y, 0.015, 'hmd')

    if l_is_valid:
        l_hand.move(l_trans, l_rot)
        l_hand.set_close_fraction(l_trig)

    vr_writer.process_frame(s)

# Note: always call this after the simulation is over to close the log file
# and clean up resources used.
vr_writer.end_log_session()
s.disconnect()