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
import pdb

import gibson2
from gibson2.objects.vr_objects import VrAgent
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.simulator import Simulator
from gibson2.task.task_base import iGTNTask
from gibson2.utils.vr_logging import VRLogWriter


import sys 
def info(type, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type, value, tb)
    else:
        traceback.print_exception(type, value, tb)
        print
        pdb.post_mortem(tb)
        


# Set to true to print out render, physics and overall frame FPS
PRINT_FPS = False

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
igtn_task = iGTNTask('assembling_gift_baskets_filtered', 0)
# igtn_task = iGTNTask('packing_lunches_filtered', 1)
# igtn_task = iGTNTask('sorting_books_filtered', 0)
# igtn_task.initialize_simulator(simulator=s, scene_id='Beechwood_0_int')
pdb.set_trace()
igtn_task.initialize_simulator(simulator=s)

kitchen_middle = [-3.7, -2.7, 1.8]

if not vr_settings.use_vr:
    camera_pose = np.array(kitchen_middle)
    view_direction = np.array([0, 1, 0])
    s.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.renderer.set_fov(90)

# Create a VrAgent and it will handle all initialization and importing under-the-hood


if vr_settings.use_vr:
    # Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the third entry of the start pos
    vr_agent = VrAgent(s)
    s.set_vr_start_pos(kitchen_middle, vr_height_offset=-0.1)


mode = 'save'

if mode == 'save':
    # Saves every 2 seconds or so (200 / 90fps is approx 2 seconds)
    vr_log_path = "./log.hdf5"
    vr_writer = VRLogWriter(frames_before_write=200, log_filepath=vr_log_path, profiling_mode=True)

    # Call set_up_data_storage once all actions have been registered (in this demo we only save states so there are none)
    # Despite having no actions, we need to call this function
    vr_writer.set_up_data_storage()

# s.optimize_vertex_and_texture()

# Main simulation loop
while True:
    s.step(print_time=PRINT_FPS)

    # Don't update VR agents or query events if we are not using VR
    if not vr_settings.use_vr:
        continue
    else:
        vr_agent.update()

    # Update VR objects
    if mode == 'save':
        vr_writer.process_frame(s)

s.disconnect()
