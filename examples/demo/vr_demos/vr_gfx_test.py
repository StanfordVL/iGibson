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
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator

# Playground configuration: edit this to change functionality
optimize = True
vr_mode = True

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
vr_rendering_settings = MeshRendererSettings(optimized=optimize,
                                            fullscreen=False,
                                            env_texture_filename=hdr_texture,
                                            env_texture_filename2=hdr_texture2,
                                            env_texture_filename3=background_texture,
                                            light_modulation_map_filename=light_modulation_map_filename,
                                            enable_shadow=False, 
                                            enable_pbr=True,
                                            msaa=True,
                                            light_dimming_factor=1.0)
# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, rendering_settings=vr_rendering_settings,
            vr_eye_tracking=False, vr_mode=vr_mode)
scene = InteractiveIndoorScene('Rs_int')
s.import_ig_scene(scene)

if not vr_mode:
    camera_pose = np.array([0, 0, 1.2])
    # Look out over the main body of the Rs scene
    view_direction = np.array([0, -1, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

if optimize:
    s.optimize_vertex_and_texture()

while True:
    s.step()

s.disconnect()