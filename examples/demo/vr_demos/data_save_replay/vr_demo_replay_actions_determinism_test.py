""" VR saving demo using simplified VR playground code.

This demo replays the actions of certain objects in the scene.

Note: This demo does not use PBR so it can be supported on a wide range of devices, including Mac OS.

This demo saves to vr_logs/vr_demo_save_states.h5
If you would like to replay the data, please run
vr_demo_replay using this file path as an input.

Run this demo if you would like to save your own data."""

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
# Whether we should hide a mustard bottle when the menu button is presed
hide_mustard_on_press = True

# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, 
            rendering_settings=MeshRendererSettings(optimized=optimize, fullscreen=fullscreen, enable_pbr=False),
            vr_eye_tracking=use_eye_tracking, vr_mode=False)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

# Player body is represented by a translucent blue cylinder
if enable_vr_body:
    vr_body = VrBody()
    s.import_object(vr_body)
    # Note: we don't call init_body since we will be controlling the body directly through pos/orientation actions

# The hand can either be 'right' or 'left'
# It has enough friction to pick up the basket and the mustard bottles
r_hand = VrHand(hand='right')
s.import_object(r_hand)
# This sets the hand constraints so it can move with the VR controller
r_hand.set_start_state(start_pos=[0, 0, 1.5])

l_hand = VrHand(hand='left')
s.import_object(l_hand)
# This sets the hand constraints so it can move with the VR controller
l_hand.set_start_state(start_pos=[0, 0.5, 1.5])

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

# Start user close to counter for interaction
s.set_vr_offset([-0.5, 0.0, -0.5])

# State of can hiding, toggled by a menu press
hide_mustard = False

# Modify this path to save to different files
vr_log_path = 'vr_logs/vr_demo_save_actions.h5'
vr_right_hand_action_path = 'vr_hand/right'
vr_left_hand_action_path = 'vr_hand/left'
vr_menu_button_action_path = 'vr_menu_button'
vr_body_action_path = 'vr_body'

vr_reader = VRLogReader(log_filepath=vr_log_path)

# Record mustard positions/orientations and save to a text file to test determinism
mustard_data = []

# In this demo, we feed actions into the simulator and simulate
# everything else.
while vr_reader.get_data_left_to_read():
    # We set fullReplay to false so we only simulate using actions
    vr_reader.read_frame(s, fullReplay=False)
    s.step()

    # Save the mustard positions each frame to a text file
    mustard_pos = mustard_list[0].get_position()
    mustard_orn = mustard_list[0].get_orientation()
    mustard_data.append(np.array(mustard_pos + mustard_orn))

    # Contains validity [0], trans [1-3], orn [4-7], trig_frac [8], touch coordinates (x and y) [9-10]
    vr_rh_actions = vr_reader.read_action(vr_right_hand_action_path)
    vr_lh_actions = vr_reader.read_action(vr_left_hand_action_path)
    vr_menu_state = vr_reader.read_action(vr_menu_button_action_path)
    vr_body_actions = vr_reader.read_action(vr_body_action_path)

    # Set mustard hidden state based on recorded button action
    if vr_menu_state == 1:
        s.set_hidden_state(mustard_list[2], hide=True)
    elif vr_menu_state == 0:
        s.set_hidden_state(mustard_list[2], hide=False)

    # Move VR hands
    if vr_rh_actions[0] == 1.0:
        r_hand.move(vr_rh_actions[1:4], vr_rh_actions[4:8])
        r_hand.set_close_fraction(vr_rh_actions[8])
    
    if vr_lh_actions[0] == 1.0:
        l_hand.move(vr_lh_actions[1:4], vr_lh_actions[4:8])
        l_hand.set_close_fraction(vr_lh_actions[8])

    # Move VR body
    vr_body.set_position_orientation(vr_body_actions[0:3], vr_body_actions[3:7])

    # Get stored eye tracking data - this is an example of how to read values that are not actions from the VRLogReader
    eye_data = vr_reader.read_value('vr/vr_eye_tracking_data')
    is_eye_data_valid = eye_data[0]
    origin = eye_data[1:4]
    direction = eye_data[4:7]
    left_pupil_diameter = eye_data[7]
    right_pupil_diameter = eye_data[8]
    
    if is_eye_data_valid:
        # Move gaze marker based on eye tracking data
        updated_marker_pos = [origin[0] + direction[0], origin[1] + direction[1], origin[2] + direction[2]]
        gaze_marker.set_position(updated_marker_pos)

    print('Mustard data information:')
    print('Length of array: {}'.format(len(mustard_data)))
    print('First element: {}'.format(mustard_data[0]))

# We always need to call end_log_session() at the end of a VRLogReader session
vr_reader.end_log_session()