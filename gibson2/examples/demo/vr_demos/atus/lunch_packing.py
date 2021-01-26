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
import signal
import sys

# Set to false to load entire Rs_int scene
LOAD_PARTIAL = False
# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = True

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
scene = InteractiveIndoorScene('Beechwood_0_int')

# Turn this on when debugging to speed up loading
if LOAD_PARTIAL:
    scene._set_first_n_objects(10)
    # Set gravity to 0 to stop all objects falling on the floor
    p.setGravity(0, 0, 0)
s.import_ig_scene(scene)

# Position that is roughly in the middle of the kitchen - used to help place objects
kitchen_middle = [-3.7, -2.7, 1.8]

if not vr_settings.use_vr:
    camera_pose = np.array(kitchen_middle)
    view_direction = np.array([0, 1, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

# Create a VrAgent and it will handle all initialization and importing under-the-hood
vr_agent = VrAgent(s, use_gripper=True)

# List of object names to filename mapping
lunch_pack_folder = os.path.join(gibson2.assets_path, 'models', 'pack_lunch')

lunch_pack_files = {
    'sandwich': os.path.join(lunch_pack_folder, 'cereal', 'cereal01', 'rigid_body.urdf'),
    'chip': os.path.join(lunch_pack_folder, 'food', 'snack', 'chips', 'chips0', 'rigid_body.urdf'),
    'fruit': os.path.join(lunch_pack_folder, 'food', 'fruit', 'pear', 'pear00', 'rigid_body.urdf'),
    'bread': os.path.join(lunch_pack_folder, 'granola', 'granola00', 'rigid_body.urdf'),
    'yogurt': os.path.join(lunch_pack_folder, 'food', 'dairy', 'yogurt', 'yogurt00_dannonbananacarton', 'rigid_body.urdf'),
    'water': os.path.join(lunch_pack_folder, 'drink', 'soda', 'soda23_mountaindew710mL', 'rigid_body.urdf'),
    'eggs': os.path.join(lunch_pack_folder, 'eggs', 'eggs00_eggland', 'rigid_body.urdf'),
    'container': os.path.join(lunch_pack_folder, 'dish', 'casserole_dish', 'casserole_dish00', 'rigid_body.urdf')
}

item_scales = {
    'sandwich': 0.7,
    'chip': 1,
    'fruit': 0.9,
    'bread': 0.7,
    'yogurt': 1,
    'water': 0.8,
    'eggs': 0.5,
    'container': 0.3
}

# A list of start positions and orientations for the objects - determined by placing objects in VR
item_start_pos_orn = {
    'sandwich': [
        [(-5.24, -1.6, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.24, -1.7, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.24, -1.8, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.24, -1.9, 0.97), (0, 0.71, 0.71, 0)],
    ],
    'chip': [
        [(-5.39, -1.62, 1.42), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.39, -1.62, 1.49), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.12, -1.62, 1.42), (-0.14, -0.06, 0.71, 0.69)],
        [(-5.12, -1.62, 1.49), (-0.14, -0.06, 0.71, 0.69)],
    ],
    'fruit': [
        [(-4.8, -3.55, 0.97), (0, 0, 0, 1)],
        [(-4.8, -3.7, 0.97), (0, 0, 0, 1)],
        [(-4.8, -3.85, 0.97), (0, 0, 0, 1)],
        [(-4.8, -4.0, 0.97), (0, 0, 0, 1)],
    ],
    'bread': [
        [(-5.39, -1.6, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.39, -1.7, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.39, -1.8, 0.97), (0, 0.71, 0.71, 0)],
        [(-5.39, -1.9, 0.97), (0, 0.71, 0.71, 0)],
    ],
    'yogurt': [
        [(-5.43, -1.64, 1.68), (0.57, 0.42, 0.42, 0.57)],
        [(-5.32, -1.64, 1.68), (0.57, 0.42, 0.42, 0.57)],
        [(-5.2, -1.64, 1.68), (0.57, 0.42, 0.42, 0.57)],
        [(-5.1, -1.64, 1.68), (0.57, 0.42, 0.42, 0.57)],
    ],
    'water': [
        [(-4.61, -1.69, 1.73), (0.68, -0.18, -0.18, 0.68)],
        [(-4.69, -1.69, 1.73), (0.68, -0.18, -0.18, 0.68)],
        [(-4.8, -1.69, 1.73), (0.68, -0.18, -0.18, 0.68)],
        [(-4.9, -1.69, 1.73), (0.68, -0.18, -0.18, 0.68)],
    ],
    'eggs': [
        [(-4.65, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.66, -1.58, 1.46), (0.72, 0, 0, 0.71)],
        [(-4.89, -1.58, 1.40), (0.72, 0, 0, 0.71)],
        [(-4.89, -1.58, 1.46), (0.72, 0, 0, 0.71)],
    ],
    'container': [
        [(-4.1, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.4, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-4.7, -1.82, 0.87), (0.71, 0, 0, 0.71)],
        [(-5.0, -1.82, 0.87), (0.71, 0, 0, 0.71)],
    ]
}



# Import all objects and put them in the correct positions
pack_items = list(lunch_pack_files.keys())
for item in pack_items:
    fpath = lunch_pack_files[item]
    start_pos_orn = item_start_pos_orn[item]
    item_scale = item_scales[item]
    for pos, orn in start_pos_orn:
        item_ob = ArticulatedObject(fpath, scale=item_scale)
        s.import_object(item_ob)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

s.optimize_vertex_and_texture()

if vr_settings.use_vr:
    # Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the third entry of the start pos
    s.set_vr_start_pos(kitchen_middle, vr_height_offset=-0.1)

# Main simulation loop
while True:
    s.step(print_time=PRINT_FPS)

    # Don't update VR agents or query events if we are not using VR
    if not vr_settings.use_vr:
        continue

    # Update VR objects
    vr_agent.update()

s.disconnect()