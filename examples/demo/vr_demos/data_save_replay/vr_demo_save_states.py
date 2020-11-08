""" VR saving demo using simplified VR playground code.

This demo saves the states of all objects in their entirety. The replay
resulting from this is completely controlled by the saved state data, and does
not involve any meaningful physical simulation.

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
from gibson2.utils.vr_logging import VRLogWriter
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
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, 
            rendering_settings=MeshRendererSettings(optimized=optimize, fullscreen=fullscreen, enable_pbr=False),
            vr_eye_tracking=use_eye_tracking, vr_mode=True)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

# Player body is represented by a translucent blue cylinder
if enable_vr_body:
    vr_body = VrBody()
    s.import_object(vr_body)
    vr_body.init_body([0,0])

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

# Modify this path to save to different files
vr_log_path = 'vr_logs/vr_demo_save_states.h5'
# Saves every 2 seconds or so (200 / 90fps is approx 2 seconds)
vr_writer = VRLogWriter(frames_before_write=200, log_filepath=vr_log_path, profiling_mode=True)

# Call set_up_data_storage once all actions have been registered (in this demo we only save states so there are none)
# Despite having no actions, we need to call this function
vr_writer.set_up_data_storage()

# Main simulation loop
for i in range(3000):
    # Step the simulator - this needs to be done every frame to actually run the simulation
    s.step()

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

        if enable_vr_body:
            # See VrBody class for more details on this method
            vr_body.move_body(s, r_touch_x, r_touch_y, movement_speed, relative_movement_device)
        else:
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

    # Record this frame's data in the VRLogWriter
    vr_writer.process_frame(s)

# Note: always call this after the simulation is over to close the log file
# and clean up resources used.
vr_writer.end_log_session()
s.disconnect()