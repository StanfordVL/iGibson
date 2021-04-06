"""
Main BEHAVIOR demo replay entrypoint
"""

import argparse
import os
import datetime
import h5py

import gibson2
from gibson2.objects.vr_objects import VrAgent
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrConditionSwitcher, VrSettings
from gibson2.simulator import Simulator
from gibson2.task.task_base import iGTNTask
from gibson2.utils.vr_logging import VRLogReader
from gibson2.utils.vr_logging import VRLogWriter
import tasknet

import numpy as np
import pybullet as p

def parse_args():
    scene_choices = [
        "Beechwood_0_int",
        "Beechwood_1_int",
        "Benevolence_0_int",
        "Benevolence_1_int",
        "Benevolence_2_int",
        "Ihlen_0_int",
        "Ihlen_1_int",
        "Merom_0_int",
        "Merom_1_int",
        "Pomaria_0_int",
        "Pomaria_1_int",
        "Pomaria_2_int",
        "Rs_int",
        "Wainscott_0_int",
        "Wainscott_1_int",
    ]

    task_choices = [
        "packing_lunches_filtered",
        "assembling_gift_baskets_filtered",
        "organizing_school_stuff_filtered",
        "re-shelving_library_books_filtered",
        "serving_hors_d_oeuvres_filtered",
        "putting_away_toys_filtered",
        "putting_away_Christmas_decorations_filtered",
        "putting_dishes_away_after_cleaning_filtered",
        "cleaning_out_drawers_filtered",
    ]
    task_id_choices = [0, 1]
    parser = argparse.ArgumentParser(
        description='Run and collect an ATUS demo')
    parser.add_argument('--vr_log_path', type=str,
                        help='Path (and filename) of vr log to replay')
    parser.add_argument('--vr_replay_log_path', type=str,
                        help='Path (and filename) of file to save replay to (for debugging)')
    parser.add_argument('--frame_save_path', type=str,
                        help='Path to save frames (frame number added automatically, as well as .jpg extension)')
    parser.add_argument('--disable_scene_cache', action='store_true',
                        help='Whether to disable using pre-initialized scene caches.')
    parser.add_argument('--disable_save',
                        action='store_true', help='Whether to disable saving log of replayed trajectory.')
    parser.add_argument('--highlight_gaze', action='store_true',
                        help='Whether to highlight the object at gaze location.')
    parser.add_argument('--profile', action='store_true',
                        help='Whether to print profiling data.')
    return parser.parse_args()


def main():
    args = parse_args()
    tasknet.set_backend("iGibson")

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
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        env_texture_filename=hdr_texture,
        env_texture_filename2=hdr_texture2,
        env_texture_filename3=background_texture,
        light_modulation_map_filename=light_modulation_map_filename,
        enable_shadow=True,
        enable_pbr=True,
        msaa=False,
        light_dimming_factor=1.0
    )

    # Initialize settings to save action replay frames
    vr_replay_settings = VrSettings()
    vr_replay_settings.turn_off_vr_mode()
    vr_replay_settings.set_frame_save_path(args.frame_save_path)
    vr_replay_settings.use_companion_window = False

    f = h5py.File(args.vr_log_path, 'r')
    task = f.attrs['/metadata/task_name']
    task_id = f.attrs['/metadata/task_instance']
    scene = f.attrs['/metadata/scene_id']

    if 'metadata/filter_objects' in f.attrs:
        filter_objects = f.attrs['metadata/filter_objects']
    else:
        filter_objects = True

    # VR system settings
    s = Simulator(mode='vr', rendering_settings=vr_rendering_settings,
                  vr_settings=vr_replay_settings)
    igtn_task = iGTNTask(task, task_id)

    scene_kwargs = None

    online_sampling = True
    if not args.disable_scene_cache:
        scene_kwargs = {
            'urdf_file': '{}_task_{}_{}_0_fixed_furniture'.format(scene, task, task_id),
        }
        online_sampling = False

    igtn_task.initialize_simulator(simulator=s, scene_id=scene, load_clutter=True,
                                   scene_kwargs=scene_kwargs, online_sampling=online_sampling)

    vr_agent = VrAgent(igtn_task.simulator)

    if not args.vr_log_path:
        raise RuntimeError('Must provide a VR log path to run action replay!')
    vr_reader = VRLogReader(args.vr_log_path, s,
                            emulate_save_fps=False, log_status=False)

    if not args.disable_save:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if args.vr_replay_log_path == None:
            args.vr_replay_log_path = "{}_{}_{}_{}.hdf5".format(task, task_id, scene, timestamp)
        vr_writer = VRLogWriter(s, igtn_task, vr_agent, frames_before_write=200, log_filepath=args.vr_replay_log_path, profiling_mode=args.profile, filter_objects=filter_objects)
        vr_writer.set_up_data_storage()


    disallowed_categories = ['walls', 'floors', 'ceilings']
    target_obj = -1
    gaze_max_distance = 100.0
    satisfied_predicates_cached = {}
    while vr_reader.get_data_left_to_read():

        if args.highlight_gaze:
            if vr_agent.vr_dict['gaze_marker'].eye_data_valid:
                if target_obj in s.scene.objects_by_id:
                    s.scene.objects_by_id[target_obj].unhighlight()

                origin = vr_agent.vr_dict['gaze_marker'].position_vector
                direction = vr_agent.vr_dict['gaze_marker'].orientation_vector
                intersection = p.rayTest(origin, np.array(origin) + (np.array(direction) * gaze_max_distance))
                target_obj = intersection[0][0]

                if target_obj in s.scene.objects_by_id:
                    obj = s.scene.objects_by_id[target_obj]
                    if obj.category not in disallowed_categories:
                        obj.highlight()

        vr_reader.pre_step()
        igtn_task.simulator.step(
            print_stats=args.profile, forced_timestep=vr_reader.get_phys_step_n())
        task_done, satisfied_predicates = igtn_task.check_success()

        # TODO: If this doesn't work, set full_replay to True to guarantee we see the same thing that the VR users saw
        vr_reader.read_frame(s, full_replay=False, print_vr_data=False)

        # Get relevant VR action data and update VR agent
        vr_action_data = vr_reader.get_vr_action_data()
        vr_agent.update(vr_action_data)

        if satisfied_predicates != satisfied_predicates_cached:
            satisfied_predicates_cached = satisfied_predicates

        if not args.disable_save:
            vr_writer.process_frame(s, store_vr_data=False)

    s.disconnect()


if __name__ == "__main__":
    main()
