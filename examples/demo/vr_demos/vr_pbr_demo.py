""" VR demo in a highly realistic PBR environment."""

import numpy as np
import os
import pybullet as p

from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.objects.vr_objects import VrHand
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import translate_vr_position_by_vecs
from gibson2 import assets_path
import gibson2

"""
use_fisheye=False,
                 msaa=False,
                 enable_shadow=False,
                 enable_pbr=True,
                 env_texture_filename=os.path.join(gibson2.ig_dataset_path, 'scenes', 'background', 'photo_studio_01_2k.hdr'),
                 env_texture_filename2=os.path.join(gibson2.ig_dataset_path, 'scenes','background', 'photo_studio_01_2k.hdr'),
                 env_texture_filename3=os.path.join(gibson2.ig_dataset_path, 'scenes', 'background', 'photo_studio_01_2k.hdr'),
                 light_modulation_map_filename='',
                 optimized=False,
                 skybox_size=20.,
                 light_dimming_factor=1.0,
                 fullscreen=False,
                 glfw_gl_version=None,
                 """

# Playground configuration: edit this to change functionality
optimize = True
vr_mode = True

# Initialize simulator with specific rendering settings
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, 
            rendering_settings=MeshRendererSettings(optimized=optimize),
            vrEyeTracking=False, vrMode=vr_mode)
scene = InteractiveIndoorScene('Rs_int')
scene._set_first_n_objects(10)
s.import_scene(scene)

camera_pose = np.array([0, 0, 1.2])
view_direction = np.array([1, 0, 0])
s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
s.renderer.set_fov(90)

if optimize:
    s.optimize_vertex_and_texture()

while True:
    s.step()

s.disconnect()