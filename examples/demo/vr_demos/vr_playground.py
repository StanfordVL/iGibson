""" VR playground containing various objects and VR options that can be toggled
to experiment with the VR experience in iGibson.

All VR functions can be found in the Simulator class.
The Simulator makes use of a MeshRendererVR, which is based
off of a VRRendererContext in render/cpp/vr_mesh_renderer.h"""

import numpy as np
import os
import pybullet as p

from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import translate_vr_position_by_vecs
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# Playground configuration: edit this to change functionality
optimize = True
vr_mode = True
print_fps = False
# Toggles fullscreen companion window
fullscreen = False
# Toggles SRAnipal eye tracking
use_eye_tracking = True
# Toggles movement with the touchpad (to move outside of play area)
touchpad_movement = True
# Set to one of hmd, right_controller or left_controller to move relative to that device
relative_movement_device = 'hmd'
# Movement speed for touchpad movement
movement_speed = 0.01

# Initialize simulator
s = Simulator(mode='vr', timestep = 1/90.0, optimized_renderer=optimize, vrFullscreen=fullscreen, vrEyeTracking=use_eye_tracking, vrMode=vr_mode)
scene = StaticIndoorScene('Placida')
s.import_scene(scene)

# This playground only uses one hand - it has enough friction to pick up some of the
# mustard bottles
rHand = VrHand()
s.import_articulated_object(rHand)
# This sets the hand constraints so it can move with the VR controller
rHand.set_start_state(start_pos=[0.0, 0.5, 1.5])

# Add playground objects to the scene
# Eye tracking visual marker - a red marker appears in the scene to indicate gaze direction
gaze_marker = VisualMarker(radius=0.03)
s.import_object(gaze_marker)
gaze_marker.set_position([0,0,1.5])

basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path)
s.import_articulated_object(basket)
basket.set_position([1, 0.2, 1])
p.changeDynamics(basket.body_id, -1, mass=5)

mass_list = [5, 10, 100, 500]
mustard_start = [1, -0.2, 1]
for i in range(len(mass_list)):
    mustard = YCBObject('006_mustard_bottle')
    s.import_articulated_object(mustard)
    mustard.set_position([mustard_start[0], mustard_start[1] - i * 0.2, mustard_start[2]])
    p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

if optimize:
    s.optimize_data()

# Start user close to counter for interaction
s.setVROffset([1.0, 0, -0.4])

# Main simulation loop
while True:
    # Demonstrates how to call VR events - replace pass with custom logic
    # See pollVREvents description in simulator for full list of events
    eventList = s.pollVREvents()
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

    # Optionally print fps during simulator step
    s.step(shouldPrintTime=print_fps)

    # VR device data
    hmdIsValid, hmdTrans, hmdRot = s.getDataForVRDevice('hmd')
    lIsValid, lTrans, lRot = s.getDataForVRDevice('left_controller')
    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')

    # VR button data
    lTrig, lTouchX, lTouchY = s.getButtonDataForController('left_controller')
    rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

    # VR eye tracking data
    is_eye_data_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.getEyeTrackingData()
    if is_eye_data_valid:
        # Move gaze marker based on eye tracking data
        updated_marker_pos = [origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]]
        gaze_marker.set_position(updated_marker_pos)

    # Get coordinate system for relative movement device
    right, _, forward = s.getDeviceCoordinateSystem(relative_movement_device)

    if rIsValid:
        rHand.move(rTrans, rRot)
        rHand.set_close_fraction(rTrig)

        # Right hand used to control movement
        # Move VR system based on device coordinate system and touchpad press location
        s.setVROffset(translate_vr_position_by_vecs(rTouchX, rTouchY, right, forward, s.getVROffset(), movement_speed))

        # Trigger haptic pulse on right touchpad, modulated by trigger close fraction
        # Close the trigger to create a stronger pulse
        # Note: open trigger has closed fraction of 0.05 when open, so cutoff haptic input under 0.1
        # to avoid constant rumbling
        s.triggerHapticPulse('right_controller', rTrig if rTrig > 0.1 else 0)

s.disconnect()