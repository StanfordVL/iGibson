""" Demo for Karen - scenario 2 - REPLAY:

Instructions:
1) Walk from A to B
2) Pick up pepper bottle at B
3) Go to C and release pepper bottle

A = far end of the room
B = kitchen bar with chairs
C = stove top with another object
"""

import numpy as np
import os
import pybullet as p
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.object_base import Object
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrBody, VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_logging import VRLogReader
from gibson2.utils.vr_utils import move_player_no_body
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')
groceries_folder = os.path.join(assets_path, 'models', 'groceries')

# HDR files for PBR rendering
hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

# TODO for Jiangshan: set this to 1, 2 or 3 depending on which trial you want to view
TRIAL_NUM = 1

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
# Initialize simulator with specific rendering settings
s = Simulator(mode='simple', image_width=504, image_height=560, rendering_settings=vr_rendering_settings)
scene = InteractiveIndoorScene('Rs_int')
s.import_ig_scene(scene)

r_hand = VrHand(hand='right')
s.import_object(r_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)

l_hand = VrHand(hand='left')
s.import_object(l_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)

gaze_marker = VisualMarker(radius=0.03)
s.import_object(gaze_marker, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
gaze_marker.set_position([0,0,1.5])

basket_path = os.path.join(sample_urdf_folder, 'object_ZU6u5fvE8Z1.urdf')
basket = ArticulatedObject(basket_path, scale=0.8)
s.import_object(basket)
basket.set_position([-1.8278704545622642, 2.152284546319316, 1.031713969848457])
p.changeDynamics(basket.body_id, -1, mass=5)

can_1_path = os.path.join(groceries_folder, 'canned_food', '1', 'rigid_body.urdf')
can_pos = [-0.8, 1.55, 1.1]
can_1 = ArticulatedObject(can_1_path, scale=0.6)
s.import_object(can_1)
can_1.set_position(can_pos)

s.optimize_vertex_and_texture()

# Note: the VRLogReader plays back the demo at the recorded fps, so there is not need to set this
vr_log_path = 'data_logs/karen_demo_scenario_2_trial_{}.h5'.format(TRIAL_NUM)
vr_reader = VRLogReader(log_filepath=vr_log_path)

# The VR reader automatically shuts itself down and performs cleanup once the while loop has finished running
while vr_reader.get_data_left_to_read():
    # We need to read frame before step for various reasons - one of them is that we need to set the camera
    # matrix for this frame before rendering in step
    vr_reader.read_frame(s, fullReplay=True)
    s.step()