""" VR embodiment demo with Fetch robot. This demo allows you to test out Fetch VR's grasping functionality."""

import numpy as np
import os
import pybullet as p
import pybullet_data

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

fetch_config = parse_config(os.path.join('..', '..', '..', 'configs', 'fetch_reaching.yaml'))

# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = True
# Set to false to just use FetchVR in non-VR mode
VR_MODE = True

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
scene._set_first_n_objects(2)
s.import_ig_scene(scene)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

if not VR_MODE:
    camera_pose = np.array([0, -3, 1.2])
    view_direction = np.array([0, 1, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

# Import FetchVR robot - the class handles importing and setup itself
fvr = FetchVR(fetch_config, s, [0.1, 0, 0], update_freq=1, use_ns_ik=True, use_gaze_marker=True)

objects = [
    ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
               0.707107)),
    ("jenga/jenga.urdf", (1.200000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
               0.707107)),
    ("jenga/jenga.urdf", (1.100000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
               0.707107)),
    ("jenga/jenga.urdf", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
               0.707107)),
    ("jenga/jenga.urdf", (0.900000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
               0.707107)),
    ("jenga/jenga.urdf", (0.800000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
               0.707107)),
    ("table/table.urdf", (1.000000, -0.200000, 0.000000), (0.000000, 0.000000, 0.707107,
               0.707107)),
    ("duck_vhacd.urdf", (1.050000, -0.500000, 0.700000), (0.000000, 0.000000, 0.707107,
               0.707107)),
    ("duck_vhacd.urdf", (0.950000, -0.100000, 0.700000), (0.000000, 0.000000, 0.707107,
               0.707107)),
    ("sphere_small.urdf", (0.850000, -0.400000, 0.700000), (0.000000, 0.000000, 0.707107,
               0.707107)),
    ("duck_vhacd.urdf", (0.850000, -0.400000, 1.00000), (0.000000, 0.000000, 0.707107,
               0.707107)),
]

for item in objects:
    fpath = item[0]
    pos = item[1]
    orn = item[2]
    item_ob = ArticulatedObject(fpath, scale=1)
    s.import_object(item_ob, use_pbr=False, use_pbr_mapping=False)
    item_ob.set_position(pos)
    item_ob.set_orientation(orn)

for i in range(3):
    obj = YCBObject('003_cracker_box')
    s.import_object(obj)
    obj.set_position_orientation([1.100000 + 0.12 * i, -0.300000, 0.750000], [0, 0, 0, 1])

obj = ArticulatedObject(os.path.join(gibson2.ig_dataset_path, 'objects', 
    'basket', 'e3bae8da192ab3d4a17ae19fa77775ff', 'e3bae8da192ab3d4a17ae19fa77775ff.urdf'),
                        scale=2)
s.import_object(obj)
p.changeDynamics(obj.body_id, -1, mass=100, lateralFriction=2)
obj.set_position_orientation([1., 0.300000, 0.750000], [0, 0, 0, 1])

s.optimize_vertex_and_texture()

# Main simulation loop
while True:
    s.step()
    if VR_MODE:
        # FetchVR class handles all update logic
        fvr.update()

s.disconnect()