"""
Main BEHAVIOR demo collection entrypoint
"""

import argparse
import datetime
import os

import numpy as np
import tasknet

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrConditionSwitcher, VrSettings
from gibson2.simulator import Simulator
from gibson2.task.task_base import iGTNTask
from gibson2.utils.ig_logging import IGLogWriter


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

    task_id_choices = [0, 1]
    parser = argparse.ArgumentParser(
        description='Run and collect an ATUS demo')
    parser.add_argument('--task', type=str, required=True,
                        nargs='?', help='Name of ATUS task matching PDDL parent folder in tasknet.')
    parser.add_argument('--task_id', type=int, required=True, choices=task_id_choices,
                        nargs='?', help='PDDL integer ID, matching suffix of pddl.')
    parser.add_argument('--vr_log_path', type=str,
                        help='Path (and filename) of vr log')
    parser.add_argument('--scene', type=str, choices=scene_choices, nargs='?',
                        help='Scene name/ID matching iGibson interactive scenes.')
    parser.add_argument('--disable_save', action='store_true',
                        help='Whether to disable saving logfiles.')
    parser.add_argument('--disable_scene_cache', action='store_true',
                        help='Whether to disable using pre-initialized scene caches.')
    parser.add_argument('--profile', action='store_true',
                        help='Whether to print profiling data.')
    parser.add_argument('--no_vr', action='store_true',
                        help='Whether to turn off VR recording and save random actions.')
    parser.add_argument('--max_steps', type=int, default=-1,
                        help="Maximum number of steps to record before stopping.")
    return parser.parse_args()


def main():
    args = parse_args()
    tasknet.set_backend("iGibson")
    collect_demo(args.task, args.task_id, args.scene, args.vr_log_path, args.disable_save, args.max_steps, args.no_vr,
                 args.disable_scene_cache, args.profile)


def collect_demo(task, task_id, scene, vr_log_path=None, disable_save=False, max_steps=-1, no_vr=False,
                 disable_scene_cache=False, profile=False):
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

    # VR system settings
    mode = 'headless' if no_vr else 'vr'
    s = Simulator(mode=mode, rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True),
                  physics_timestep=1 / 300.0, render_timestep=1 / 30.0)
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

    if not no_vr:
        vr_cs = VrConditionSwitcher(
            igtn_task.simulator,
            igtn_task.show_instruction,
            igtn_task.iterate_instruction
        )

    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if vr_log_path is None:
            vr_log_path = "{}_{}_{}_{}.hdf5".format(
                task, task_id, scene, timestamp)
        log_writer = IGLogWriter(
            s,
            frames_before_write=200,
            log_filepath=vr_log_path,
            task=igtn_task,
            store_vr=False if no_vr else True,
            vr_robot=vr_agent,
            profiling_mode=profile,
            filter_objects=True
        )
        log_writer.set_up_data_storage()

    satisfied_predicates_cached = {}
    post_task_steps = 200
    physics_warming_steps = 200

    steps = 0
    while max_steps < 0 or steps < max_steps:
        igtn_task.simulator.step(print_stats=profile)
        task_done, satisfied_predicates = igtn_task.check_success()

        if no_vr:
            if steps < 2:
                action = np.zeros((28,))
                action[19] = 1
                action[27] = 1
            else:
                action = np.random.uniform(-0.01, 0.01, size=(28,))
        else:
            action = igtn_task.simulator.gen_vr_robot_action()
            if steps < physics_warming_steps:
                action = np.zeros_like(action)

        vr_agent.update(action)

        if not no_vr:
            if satisfied_predicates != satisfied_predicates_cached:
                vr_cs.refresh_condition(switch=False)
                satisfied_predicates_cached = satisfied_predicates

            if igtn_task.simulator.query_vr_event('right_controller', 'overlay_toggle'):
                vr_cs.refresh_condition()

            if igtn_task.simulator.query_vr_event('left_controller', 'overlay_toggle'):
                vr_cs.toggle_show_state()

        if log_writer and not disable_save:
            log_writer.process_frame()

        if task_done:
            post_task_steps -= 1
            if post_task_steps == 0:
                break

        steps += 1

    if log_writer and not disable_save:
        log_writer.end_log_session()

    s.disconnect()


if __name__ == "__main__":
    main()
