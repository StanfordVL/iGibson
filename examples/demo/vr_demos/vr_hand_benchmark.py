""" This is a simple object picking and placing task
that can be used to benchmark the dexterity of the VR hand.
"""
import numpy as np
import os
import pybullet as p
import pybullet_data
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.object_base import Object
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2 import assets_path
import signal
import sys

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
# TODO: Add option for gui benchmarking vs vr benchmarking
s = Simulator(mode='iggui', 
              image_width=512,
              image_height=512,
              rendering_settings=vr_rendering_settings, 
              )

scene = InteractiveIndoorScene('Rs_int')
# Turn this on when debugging to speed up loading
scene._set_first_n_objects(2)
s.import_ig_scene(scene)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

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
obj.set_position_orientation([1., 0.300000, 0.750000], [0, 0, 0, 1])

# Main simulation loop
while True:
    s.step()
s.disconnect()