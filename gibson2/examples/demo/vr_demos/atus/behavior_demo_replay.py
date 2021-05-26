"""
Main BEHAVIOR demo replay entrypoint
"""

import argparse
import os
import datetime
import h5py
import pprint

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.simulator import Simulator
from gibson2.task.task_base import iGTNTask
from gibson2.utils.ig_logging import IGLogReader, IGLogWriter
from gibson2.utils.git_utils import project_git_info
from gibson2.utils.utils import parse_str_config
import tasknet

import numpy as np
import pybullet as p


def parse_args():
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
                        action='store_true', help='Whether to disable saving log of replayed trajectory, used for validation.')
    parser.add_argument('--highlight_gaze', action='store_true',
                        help='Whether to highlight the object at gaze location.')
    parser.add_argument('--profile', action='store_true',
                        help='Whether to print profiling data.')
    parser.add_argument('--no_vr', action='store_true',
                        help='Whether to disable replay through VR and use iggui instead.')
    return parser.parse_args()


def main():
    args = parse_args()
    tasknet.set_backend("iGibson")
    replay_demo(args.vr_log_path, args.vr_replay_log_path, args.frame_save_path, args.highlight_gaze, args.no_vr,
                args.disable_save, args.disable_scene_cache, args.profile)


def replay_demo(vr_log_path, vr_replay_log_path=None, frame_save_path=None, highlight_gaze=False, no_vr=False,
                disable_save=False, disable_scene_cache=False, profile=False):
    pp = pprint.PrettyPrinter(indent=4)

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
    vr_settings = VrSettings(config_str=IGLogReader.read_metadata_attr(vr_log_path, '/metadata/vr_settings'))
    vr_settings.set_frame_save_path(frame_save_path)

    task = IGLogReader.read_metadata_attr(vr_log_path, '/metadata/task_name')
    task_id = IGLogReader.read_metadata_attr(vr_log_path, '/metadata/task_instance')
    scene = IGLogReader.read_metadata_attr(vr_log_path, '/metadata/scene_id')
    physics_timestep = IGLogReader.read_metadata_attr(vr_log_path, '/metadata/physics_timestep')
    render_timestep = IGLogReader.read_metadata_attr(vr_log_path, '/metadata/render_timestep')

    if IGLogReader.has_metadata_attr(vr_log_path, '/metadata/filter_objects'):
        filter_objects = IGLogReader.read_metadata_attr(vr_log_path, '/metadata/filter_objects')
    else:
        filter_objects = True

    if IGLogReader.has_metadata_attr(vr_log_path, '/metadata/git_info'):
        logged_git_info = IGLogReader.read_metadata_attr(vr_log_path, '/metadata/git_info')
        logged_git_info = parse_str_config(logged_git_info)
        git_info = project_git_info()
        for key in logged_git_info.keys():
            logged_git_info[key].pop('directory', None)
            git_info[key].pop('directory', None)
            if logged_git_info[key] != git_info[key]:
                print("Warning, difference in git commits for repo: {}. This may impact deterministic replay".format(key))
                print("Logged git info:\n")
                pp.pprint(logged_git_info[key])
                print("Current git info:\n")
                pp.pprint(git_info[key])

    # Get dictionary mapping object body id to name, also check it is a dictionary
    obj_body_id_to_name = IGLogReader.get_obj_body_id_to_name(vr_log_path)
    assert type(obj_body_id_to_name) == dict

    # VR system settings
    s = Simulator(
          mode='simple' if no_vr else 'vr',
          physics_timestep = physics_timestep,
          render_timestep = render_timestep,
          rendering_settings=vr_rendering_settings,
          vr_settings=vr_settings,
          image_width=1280,
          image_height=720,
        )

    igtn_task = iGTNTask(task, task_id)

    scene_kwargs = None

    online_sampling = True

    if not disable_scene_cache:
        scene_kwargs = {
            'urdf_file': '{}_neurips_task_{}_{}_0_fixed_furniture'.format(scene, task, task_id),
        }
        online_sampling = False

    igtn_task.initialize_simulator(simulator=s,
                                   scene_id=scene,
                                   scene_kwargs=scene_kwargs,
                                   load_clutter=True,
                                   online_sampling=online_sampling)
    vr_agent = igtn_task.simulator.robots[0]
    if not vr_log_path:
        raise RuntimeError('Must provide a VR log path to run action replay!')
    log_reader = IGLogReader(vr_log_path, log_status=False)

    if not disable_save:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if vr_replay_log_path == None:
            vr_replay_log_path = "{}_{}_{}_{}.hdf5".format(
                task, task_id, scene, timestamp)

        replay_path = vr_log_path[:-5] + "_replay.hdf5"
        log_writer = IGLogWriter(
            s,
            frames_before_write=200,
            log_filepath=replay_path,
            task=igtn_task,
            store_vr=False,
            vr_robot=vr_agent,
            profiling_mode=profile,
            filter_objects=filter_objects
        )
        log_writer.set_up_data_storage()

    disallowed_categories = ['walls', 'floors', 'ceilings']
    target_obj = -1
    gaze_max_distance = 100.0
    task_done = False
    satisfied_predicates_cached = {}
    while log_reader.get_data_left_to_read():
        if highlight_gaze:
            eye_data = log_reader.get_vr_data().query('eye_data')
            if eye_data[0]:
                if target_obj in s.scene.objects_by_id:
                    s.scene.objects_by_id[target_obj].unhighlight()

                origin = eye_data[1]
                direction = eye_data[2]
                intersection = p.rayTest(origin, np.array(
                    origin) + (np.array(direction) * gaze_max_distance))
                target_obj = intersection[0][0]

                if target_obj in s.scene.objects_by_id:
                    obj = s.scene.objects_by_id[target_obj]
                    if obj.category not in disallowed_categories:
                        obj.highlight()

        igtn_task.simulator.step(print_stats=profile)
        task_done, satisfied_predicates = igtn_task.check_success()

        # Set camera each frame
        if not no_vr:
            log_reader.set_replay_camera(s)

        # Get relevant VR action data and update VR agent
        vr_agent.update(log_reader.get_agent_action('vr_robot'))

        if satisfied_predicates != satisfied_predicates_cached:
            satisfied_predicates_cached = satisfied_predicates

        if not disable_save:
            log_writer.process_frame()

        # Per-step determinism check. Activate if necessary.
        # things_to_compare = [thing for thing in log_writer.name_path_data if thing[0] == "physics_data"]
        # for thing in things_to_compare:
        #     thing_path = "/".join(thing)
        #     fc = log_reader.frame_counter % log_writer.frames_before_write
        #     if fc == log_writer.frames_before_write - 1:
        #         continue
        #     replayed = log_writer.get_data_for_name_path(thing)[fc]
        #     original = log_reader.read_value(thing_path)
        #     if not np.all(replayed == original):
        #         print("%s not equal in %d" % (thing_path, log_reader.frame_counter))
        #     if not np.isclose(replayed, original).all():
        #         print("%s not close in %d" % (thing_path, log_reader.frame_counter))

    print("Demo was succesfully completed: ", task_done)

    is_deterministic = False
    if not disable_save:
        log_writer.end_log_session()
        with h5py.File(vr_log_path) as original_file, h5py.File(replay_path) as new_file:
            is_deterministic = True
            for obj in new_file['physics_data']:
                for attribute in new_file['physics_data'][obj]:
                    is_close = np.isclose(original_file['physics_data'][obj][attribute], new_file['physics_data'][obj][attribute]).all()
                    is_deterministic = is_deterministic and is_close
                    if not is_close:
                        print("Mismatch for obj {} with mismatched attribute {}".format(obj, attribute))

        print("Demo was deterministic: ", is_deterministic)
    s.disconnect()

    return is_deterministic


if __name__ == "__main__":
    main()
