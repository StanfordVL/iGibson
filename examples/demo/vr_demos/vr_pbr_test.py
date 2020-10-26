""" VR playground containing various objects and VR options that can be toggled
to experiment with the VR experience in iGibson. This playground operates in a
PBR scene. Please see vr_playground_no_pbr.py for a non-PBR experience.

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
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import move_player_no_body
from gibson2 import assets_path
import gibson2
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# Playground configuration: edit this to change functionality
optimize = True
vr_mode = False
# Toggles fullscreen companion window
fullscreen = False
# Toggles SRAnipal eye tracking
use_eye_tracking = True
# Enables the VR collision body
enable_vr_body = False
# Toggles movement with the touchpad (to move outside of play area)
touchpad_movement = True
# Set to one of hmd, right_controller or left_controller to move relative to that device
relative_movement_device = 'hmd'
# Movement speed for touchpad-based movement
movement_speed = 0.02

hdr_texture = os.path.join(
        gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

rendering_settings = MeshRendererSettings(optimized=True,
                                    env_texture_filename=hdr_texture,
                                    env_texture_filename2=hdr_texture2,
                                    env_texture_filename3=background_texture,
                                    light_modulation_map_filename=light_modulation_map_filename,
                                    enable_pbr=True,
                                    enable_shadow=False, 
                                    msaa=True,
                                    fullscreen=fullscreen,
                                    light_dimming_factor=1.0)

# Initialize simulator with specific rendering settings
s = Simulator(mode='iggui', physics_timestep = 1/90.0, render_timestep = 1/90.0, image_width=512, image_height=512,
            rendering_settings=rendering_settings, vrMode=vr_mode)
scene = InteractiveIndoorScene('Beechwood_0_int')
#scene._set_first_n_objects(40)
s.import_scene(scene)

#camera_pose = np.array([0, 0, 1.2])
#view_direction = np.array([1, 0, 0])
#s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
#s.renderer.set_fov(90)

# Player body is represented by a translucent blue cylinder
if enable_vr_body:
    vr_body = VrBody()
    s.import_object(vr_body)
    vr_body.init_body([0,0])

# This playground only uses one hand - it has enough friction to pick up some of the
# mustard bottles
rHand = VrHand()
s.import_object(rHand)
# This sets the hand constraints so it can move with the VR controller
rHand.set_start_state(start_pos=[0.0, 0.5, 1.5])

# Add playground objects to the scene
# Eye tracking visual marker - a red marker appears in the scene to indicate gaze direction
gaze_marker = VisualMarker(radius=0.03)
s.import_object(gaze_marker)
gaze_marker.set_position([0,0,1.5])

if optimize:
    s.optimize_vertex_and_texture()

# Start user close to counter for interaction
#s.setVROffset([-2.0, 0.0, -0.4])

while True:
    # Step the simulator - this needs to be done every frame to actually run the simulation
    s.step()

    rIsValid, rTrans, rRot = s.getDataForVRDevice('right_controller')
    rTrig, rTouchX, rTouchY = s.getButtonDataForController('right_controller')

    # VR eye tracking data
    is_eye_data_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = s.getEyeTrackingData()
    if is_eye_data_valid:
        # Move gaze marker based on eye tracking data
        updated_marker_pos = [origin[0] + dir[0], origin[1] + dir[1], origin[2] + dir[2]]
        gaze_marker.set_position(updated_marker_pos)

    if rIsValid:
        rHand.move(rTrans, rRot)
        rHand.set_close_fraction(rTrig)

        if enable_vr_body:
            # See VrBody class for more details on this method
            vr_body.move_body(s, rTouchX, rTouchY, movement_speed, relative_movement_device)
        else:
            # Right hand used to control movement
            # Move VR system based on device coordinate system and touchpad press location
            move_player_no_body(s, rTouchX, rTouchY, movement_speed, relative_movement_device)

s.disconnect()