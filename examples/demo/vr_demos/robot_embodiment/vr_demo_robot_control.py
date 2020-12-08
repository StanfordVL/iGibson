""" VR embodiment demo with Fetch robot. """

import numpy as np
import os
import pybullet as p

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.robots.fetch_vr_robot import FetchVR
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrGazeMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.utils import parse_config
from gibson2.utils.vr_utils import move_player
from gibson2 import assets_path

sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')
fetch_config = parse_config(os.path.join('..', '..', '..', 'configs', 'fetch_p2p_nav.yaml'))

# Set to false to load entire Rs_int scene
LOAD_PARTIAL = True
# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = False
# Set to false to just use FetchVR in non-VR mode
VR_MODE = False

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
if VR_MODE:
    s = Simulator(mode='vr', 
                rendering_settings=vr_rendering_settings, 
                vr_settings=VrSettings())
else:
    s = Simulator(mode='iggui', image_width=960,
                  image_height=720, device_idx=0, rendering_settings=vr_rendering_settings)
    s.viewer.min_cam_z = 1.0

scene = InteractiveIndoorScene('Rs_int')
# Turn this on when debugging to speed up loading
if LOAD_PARTIAL:
    scene._set_first_n_objects(10)
s.import_ig_scene(scene)

# Import FetchVR robot - the class handles importing and setup itself
fvr = FetchVR(fetch_config, s, [0.5, -1.5, 0], update_freq=1)

# Gaze marker to visualize where the user is looking
gm = VrGazeMarker(s)

# Objects to interact with
basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path, scale=0.8)
s.import_object(basket)
basket.set_position([-1, 1.55, 1.2])
p.changeDynamics(basket.body_id, -1, mass=5)

s.optimize_vertex_and_texture()

# Main simulation loop
while True:
    s.step()
    if VR_MODE:
        # FetchVR class handles all update logic
        fvr.update()
        # Update visual gaze marker
        gm.update()

s.disconnect()