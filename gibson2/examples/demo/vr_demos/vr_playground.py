""" VR playground containing various objects. This playground operates in the
Rs_int PBR scene.

Important - VR functionality and where to find it:

1) Most VR functions can be found in the gibson2/simulator.py
2) The VrAgent and its associated VR objects can be found in gibson2/objects/vr_objects.py
3) VR utility functions are found in gibson2/utils/vr_utils.py
4) The VR renderer can be found in gibson2/render/mesh_renderer.py
5) The underlying VR C++ code can be found in vr_mesh_render.h and .cpp in gibson2/render/cpp
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
from gibson2.objects.vr_objects import VrAgent
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path

# Set to false to load entire Rs_int scene
LOAD_PARTIAL = False
# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = True
# Set to true to use VR hand instead of gripper
USE_HAND = True

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
# VR system settings
# Change use_vr to toggle VR mode on/off
vr_settings = VrSettings(use_vr=True)
s = Simulator(mode='vr', 
            rendering_settings=vr_rendering_settings, 
            vr_settings=vr_settings)
scene = InteractiveIndoorScene('Rs_int')
# Turn this on when debugging to speed up loading
if LOAD_PARTIAL:
    scene._set_first_n_objects(10)
s.import_ig_scene(scene)

if not vr_settings.use_vr:
    camera_pose = np.array([0, -3, 1.2])
    view_direction = np.array([0, 1, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

# Create a VrAgent and it will handle all initialization and importing under-the-hood
# Change use_gripper to switch between the VrHand and the VrGripper (see objects/vr_objects.py for more details)
vr_agent = VrAgent(s, use_gripper=not USE_HAND)

# Objects to interact with
mass_list = [5, 10, 100, 500]
mustard_start = [-1, 1.55, 1.2]
for i in range(len(mass_list)):
    mustard = YCBObject('006_mustard_bottle')
    s.import_object(mustard, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
    mustard.set_position([mustard_start[0] + i * 0.2, mustard_start[1], mustard_start[2]])
    p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

s.optimize_vertex_and_texture()

if vr_settings.use_vr:
    # Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the third entry of the start pos
    s.set_vr_start_pos([0, 0, 0], vr_height_offset=-0.1)

# Main simulation loop
while True:
    s.step(print_time=PRINT_FPS)

    # Don't update VR agents or query events if we are not using VR
    if not vr_settings.use_vr:
        continue

    # Example of querying VR events to hide object
    if s.query_vr_event('right_controller', 'touchpad_press'):
        s.set_hidden_state(mustard, hide=not s.get_hidden_state(mustard))

    # Update VR objects
    vr_agent.update()

s.disconnect()