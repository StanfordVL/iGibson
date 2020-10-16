""" VR demo in a highly realistic PBR environment."""

import numpy as np
import os
import pybullet as p

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

# Playground configuration: edit this to change functionality
optimize = True
vr_mode = False

# Initialize simulator
s = Simulator(mode='vr', physics_timestep = 1/90.0, render_timestep = 1/90.0, 
            optimized_renderer=optimize, vrFullscreen=False, vrEyeTracking=False, vrMode=vr_mode)
#scene = StaticIndoorScene('Placida')
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