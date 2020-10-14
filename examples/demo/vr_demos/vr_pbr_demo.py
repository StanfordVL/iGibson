""" VR demo in a highly realistic PBR environment."""

import numpy as np
import os
import pybullet as p

from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import translate_vr_position_by_vecs
from gibson2 import assets_path
import gibson2

# Playground configuration: edit this to change functionality
optimize = True
vr_mode = True
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
s = Simulator(mode='vr', timestep = 1/90.0, optimized_renderer=optimize, vrFullscreen=fullscreen, vrEyeTracking=use_eye_tracking, vrMode=vr_mode,
            env_texture_filename=os.path.join(gibson2.assets_path, 'hdr', 'photo_studio_01_2k.hdr'))
scene = InteractiveIndoorScene('Beechwood_0')
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

if optimize:
    s.optimize_data()

# Initial offset - TODO: Get rid of this?
s.setVROffset([1.0, 0, -0.4])

while True:
    s.step(shouldPrintTime=True)
    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')
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
        s.setVROffset(translate_vr_position_by_vecs(rTouchX, rTouchY, right, forward, s.getVROffset(), movement_speed))

s.disconnect()