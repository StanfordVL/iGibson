""" Simple scene for tuning the VR body. """

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
from gibson2.utils.vr_utils import move_player
from gibson2 import assets_path
sample_urdf_folder = os.path.join(assets_path, 'models', 'sample_urdfs')
groceries_folder = os.path.join(assets_path, 'models', 'groceries')

hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

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
s = Simulator(mode='vr', rendering_settings=vr_rendering_settings, vr_eye_tracking=False, vr_mode=True)
scene = InteractiveIndoorScene('Rs_int')
scene._set_first_n_objects(5)
s.import_ig_scene(scene)

vr_body = VrBody(s)
s.import_object(vr_body, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
vr_body.init_body()

r_hand = VrHand(s, hand='right')
s.import_object(r_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
r_hand.hand_setup()

l_hand = VrHand(s, hand='left')
s.import_object(l_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
l_hand.hand_setup()

s.optimize_vertex_and_texture()

while True:
    event_list = s.poll_vr_events()
    for event in event_list:
        device_type, event_type = event
        if event_type == 'grip_press':
            if device_type == 'left_controller':
                l_hand.reset_hand_transform()
            else:
                r_hand.reset_hand_transform()

    s.step()
    l_is_valid, l_trans, l_rot = s.get_data_for_vr_device('left_controller')
    r_is_valid, r_trans, r_rot = s.get_data_for_vr_device('right_controller')
    l_trig, l_touch_x, l_touch_y = s.get_button_data_for_controller('left_controller')
    r_trig, r_touch_x, r_touch_y = s.get_button_data_for_controller('right_controller')

    if r_is_valid:
        r_hand.move(r_trans, r_rot)
        r_hand.set_close_fraction(r_trig)
        move_player(s, r_touch_x, r_touch_y, 0.01, 'hmd')

    if l_is_valid:
        l_hand.move(l_trans, l_rot)
        l_hand.set_close_fraction(l_trig)

    vr_body.update_body()

s.disconnect()