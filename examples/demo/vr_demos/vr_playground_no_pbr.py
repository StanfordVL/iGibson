""" VR playground containing various objects and VR options that can be toggled
to experiment with the VR experience in iGibson. This playground operates in
the Placida scene, which does not use PBR.

Important: VR functionality and where to find it:

1) Most VR functions can be found in the gibson2/simulator.py
2) VR utility functions are found in gibson2/utils/vr_utils.py
3) The VR renderer can be found in gibson2/render/mesh_renderer.py
4) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in gibson2/render/cpp
"""

import numpy as np
import os
import pybullet as p

from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject, VArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import move_player_no_body
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# Playground configuration: edit this to change functionality
optimize = True
vr_mode = True
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
movement_speed = 0.02

# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, 
            rendering_settings=MeshRendererSettings(optimized=optimize, fullscreen=fullscreen, enable_pbr=False),
            vr_eye_tracking=use_eye_tracking, vr_mode=vr_mode)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

# TODO: Set gravity back once I finish debugging the VR hands
#p.setGravity(0,0,0)

# Player body is represented by a translucent blue cylinder
if enable_vr_body:
    vr_body = VrBody()
    s.import_object(vr_body)
    vr_body.init_body([0,0])

# The hand can either be 'right' or 'left'
# It has enough friction to pick up the basket and the mustard bottles
#rHand = VrHand(hand='right')
#s.import_object(rHand)
# This sets the hand constraints so it can move with the VR controller
#rHand.set_start_state(start_pos=[0.0, 0.5, 1.5])
vr_hand_path = os.path.join(assets_path, 'models', 'vr_hand', 'vr_hand_right.urdf')
vr_hand_r = ArticulatedObject(vr_hand_path)
s.import_object(vr_hand_r)
vr_hand_r.set_position([-0.85, -0.6, 0.8])

# Add playground objects to the scene
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

# Hide third mustard as a test
# TODO: Figure out how to hide objects - is setting gl_position enough?
s.set_hidden_state(mustard_list[2])

# Start user close to counter for interaction
s.set_vr_offset([-0.5, 0.0, -0.4])

# TODO: Test both VR hands and add in dynamic hiding
# Main simulation loop
while True:
    # Demonstrates how to call VR events - replace pass with custom logic
    # See pollVREvents description in simulator for full list of events
    eventList = s.poll_vr_events()
    for event in eventList:
        deviceType, eventType = event
        if deviceType == 'left_controller':
            if eventType == 'trigger_press':
                pass
            elif eventType == 'trigger_unpress':
                pass
        elif deviceType == 'right_controller':
            if eventType == 'trigger_press':
                pass
            elif eventType == 'trigger_unpress':
                pass

    # Step the simulator - this needs to be done every frame to actually run the simulation
    s.step()

    # VR device data
    hmdIsValid, hmdTrans, hmdRot = s.get_data_for_vr_device('hmd')
    lIsValid, lTrans, lRot = s.get_data_for_vr_device('left_controller')
    rIsValid, rTrans, rRot = s.get_data_for_vr_device('right_controller')

    # VR button data
    lTrig, lTouchX, lTouchY = s.get_button_data_for_controller('left_controller')
    rTrig, rTouchX, rTouchY = s.get_button_data_for_controller('right_controller')

    # VR eye tracking data
    is_eye_data_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.get_eye_tracking_data()
    if is_eye_data_valid:
        # Move gaze marker based on eye tracking data
        updated_marker_pos = [origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]]
        gaze_marker.set_position(updated_marker_pos)

    if rIsValid:
        #rHand.move(rTrans, rRot)
        #rHand.set_close_fraction(rTrig)

        if enable_vr_body:
            # See VrBody class for more details on this method
            vr_body.move_body(s, rTouchX, rTouchY, movement_speed, relative_movement_device)
        else:
            # Right hand used to control movement
            # Move VR system based on device coordinate system and touchpad press location
            move_player_no_body(s, rTouchX, rTouchY, movement_speed, relative_movement_device)

        # Trigger haptic pulse on right touchpad, modulated by trigger close fraction
        # Close the trigger to create a stronger pulse
        # Note: open trigger has closed fraction of 0.05 when open, so cutoff haptic input under 0.1
        # to avoid constant rumbling
        s.trigger_haptic_pulse('right_controller', rTrig if rTrig > 0.1 else 0)

s.disconnect()