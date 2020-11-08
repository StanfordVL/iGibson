""" A very simple VR program containing only a single scene.
The user can fly around the scene using the controller, and can
explore whether all the graphics features of iGibson are working as intended.
"""

import numpy as np
import os
import pybullet as p
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator
from gibson2.utils.vr_utils import move_player_no_body

optimize = True
vr_mode = True
time_fps = True

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
# Turn this on when debugging to speed up loading
scene._set_first_n_objects(10)
s.import_ig_scene(scene)

if not vr_mode:
    camera_pose = np.array([0, 0, 1.2])
    view_direction = np.array([0, -1, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

if optimize:
    s.optimize_vertex_and_texture()

while True:
    start_time = time.time()
    s.step()

    # Allow movement to explore the scene
    r_is_valid, _, _ = s.get_data_for_vr_device('right_controller')
    _, r_touch_x, r_touch_y = s.get_button_data_for_controller('right_controller')
    if r_is_valid:
        move_player_no_body(s, r_touch_x, r_touch_y, 0.03, 'hmd')

    frame_dur = time.time() - start_time
    if time_fps:
        print('Fps: {}'.format(round(1/max(frame_dur, 0.00001), 2)))

s.disconnect()