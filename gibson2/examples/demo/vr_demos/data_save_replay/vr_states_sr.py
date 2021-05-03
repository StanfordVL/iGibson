""" VR saving/replay demo.

This demo saves the states of all objects in their entirety. The replay
resulting from this is completely controlled by the saved state data, and does
not involve any meaningful physical simulation.

Usage:
python vr_states_sr.py --mode=[save/replay]

This demo saves to vr_logs/vr_states_sr.h5
Run this demo (and also change the filename) if you would like to save your own data."""

import argparse
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
from gibson2.objects.visual_marker import VisualMarker
from gibson2.objects.ycb_object import YCBObject
from gibson2.simulator import Simulator
from gibson2.utils.vr_logging import VRLogReader, VRLogWriter
from gibson2 import assets_path

# Number of seconds to run the data saving for
DATA_SAVE_RUNTIME = 45

def run_state_sr(mode):
    """
    Runs state save/replay. Mode can either be save or replay.
    """
    assert mode in ['save', 'replay']
    is_save = (mode == 'save')

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
    vr_settings = VrSettings()
    if not is_save:
        vr_settings.turn_off_vr_mode()
    s = Simulator(mode='vr', 
                rendering_settings=vr_rendering_settings, 
                vr_settings=vr_settings)
    scene = InteractiveIndoorScene('Rs_int')
    s.import_ig_scene(scene)

    # Create a VrAgent and it will handle all initialization and importing under-the-hood
    vr_agent = VrAgent(s, use_constraints=is_save)
    if is_save:
        # Since vr_height_offset is set, we will use the VR HMD true height plus this offset instead of the third entry of the start pos
        s.set_vr_start_pos([0, 0, 0], vr_height_offset=-0.1)

    # Objects to interact with
    mass_list = [5, 10, 100, 500]
    mustard_start = [-1, 1.55, 1.2]
    for i in range(len(mass_list)):
        mustard = YCBObject('006_mustard_bottle')
        s.import_object(mustard, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
        mustard.set_position([mustard_start[0] + i * 0.2, mustard_start[1], mustard_start[2]])
        p.changeDynamics(mustard.body_id, -1, mass=mass_list[i])

    # Note: Modify this path to save to different files
    vr_log_path = 'vr_logs/vr_states_sr.h5'

    if is_save:
        # Saves every few seconds
        vr_writer = VRLogWriter(frames_before_write=200, log_filepath=vr_log_path, profiling_mode=False)

        # Call set_up_data_storage once all actions have been registered (in this demo we only save states so there are none)
        # Despite having no actions, we need to call this function
        vr_writer.set_up_data_storage()
    else:
        # Playback faster than FPS during saving - can set emulate_save_fps to True to emulate saving FPS
        vr_reader = VRLogReader(vr_log_path, s.vr_settings, emulate_save_fps=False)

    if is_save:
        start_time = time.time()
        # Main simulation loop - run for as long as the user specified
        while (time.time() - start_time < DATA_SAVE_RUNTIME):
            s.step()

            # Update VR objects
            vr_agent.update()

            # Record this frame's data in the VRLogWriter
            vr_writer.process_frame(s)

        # Note: always call this after the simulation is over to close the log file
        # and clean up resources used.
        vr_writer.end_log_session()
    else:
        # The VR reader automatically shuts itself down and performs cleanup once the while loop has finished running
        while vr_reader.get_data_left_to_read():
            vr_reader.pre_step()
            s.step(forced_timestep=vr_reader.get_phys_step_n())
            vr_reader.read_frame(s, full_replay=True)
    
    s.disconnect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='VR state saving and replay demo')
    parser.add_argument('--mode', default='save', help='Mode to run in: either save or replay')
    args = parser.parse_args()
    run_state_sr(mode=args.mode)