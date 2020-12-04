""" VR playground containing various objects. This playground operates in a
PBR scene. Please see vr_playground_no_pbr.py for a non-PBR experience.

Important - VR functionality and where to find it:

1) Most VR functions can be found in the gibson2/simulator.py
2) VR utility functions are found in gibson2/utils/vr_utils.py
3) The VR renderer can be found in gibson2/render/mesh_renderer.py
4) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in gibson2/render/cpp
"""

import numpy as np
import os
import pybullet as p
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.object_base import Object
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand, VrGazeMarker
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path

sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')

# Set to false to load entire Rs_int scene
LOAD_PARTIAL = False

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
s = Simulator(mode='vr', 
            rendering_settings=vr_rendering_settings, 
            vr_settings=VrSettings())
scene = InteractiveIndoorScene('Rs_int')
# Turn this on when debugging to speed up loading
if LOAD_PARTIAL:
    scene._set_first_n_objects(10)
s.import_ig_scene(scene)

# VR objects automatically import themselves into simulator and perform setup
# use_constraints allows the body and hands to be controlled by PyBullet's constraint system
# This is turned off during full-state data replay
vr_body = VrBody(s, use_constraints=True)
r_hand = VrHand(s, hand='right', use_constraints=True)
l_hand = VrHand(s, hand='left', use_constraints=True)
gaze_marker = VrGazeMarker(s)

# Objects to interact with
basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path, scale=0.8)
s.import_object(basket)
basket.set_position([-1, 1.55, 1.2])
p.changeDynamics(basket.body_id, -1, mass=5)

s.optimize_vertex_and_texture()

# Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the third entry of the start pos
s.set_vr_start_pos([0, 0, 0], vr_height_offset=-0.1)

# Main simulation loop
while True:
    s.step(print_time=True)

    # Example of querying VR events to hide object
    if s.query_vr_event('right_controller', 'touchpad_press'):
        s.set_hidden_state(basket, hide=not s.get_hidden_state(basket))

    # Update VR objects
    gaze_marker.update_marker()
    r_hand.update_hand()
    l_hand.update_hand()
    vr_body.update_body()

s.disconnect()