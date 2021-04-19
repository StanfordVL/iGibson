""" 
Main BEHAVIOR demo collection entrypoint
"""

import argparse
import os
import datetime

import gibson2
from gibson2.objects.vr_objects import VrAgent
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrConditionSwitcher
from gibson2.simulator import Simulator
from gibson2.task.task_base import iGTNTask
from gibson2.utils.vr_logging import VRLogWriter
import tasknet


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
    parser.add_argument('--task', type=str, required=True, choices=task_choices,
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

    # VR system settings
    s = Simulator(mode='vr', rendering_settings=vr_rendering_settings)
    igtn_task = iGTNTask(args.task, args.task_id)

    scene_kwargs = None
    online_sampling = True
    offline_sampling = False

    if not args.disable_scene_cache:
        scene_kwargs = {
            'urdf_file': '{}_task_{}_{}_0_fixed_furniture'.format(args.scene, args.task, args.task_id),
        }
        online_sampling = False
        offline_sampling = True

    igtn_task.initialize_simulator(simulator=s,
                                   scene_id=args.scene,
                                   scene_kwargs=scene_kwargs,
                                   load_clutter=True,
                                   online_sampling=online_sampling,
                                   offline_sampling=offline_sampling)

    vr_agent = VrAgent(igtn_task.simulator)
    igtn_task.simulator.set_vr_start_pos([0, 0, 1.8], vr_height_offset=-0.1)

    vr_cs = VrConditionSwitcher(
        igtn_task.simulator,
        igtn_task.show_instruction,
        igtn_task.iterate_instruction
    )

    if not args.disable_save:
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        if args.vr_log_path == None:
            args.vr_log_path = "{}_{}_{}_{}.hdf5".format(
                args.task, args.task_id, args.scene, timestamp)
        vr_writer = VRLogWriter(s, igtn_task, vr_agent, frames_before_write=200,
                                log_filepath=args.vr_log_path, profiling_mode=args.profile)
        vr_writer.set_up_data_storage()

    satisfied_predicates_cached = {}
    while True:
        igtn_task.simulator.step(print_stats=args.profile)
        task_done, satisfied_predicates = igtn_task.check_success()

        vr_agent.update()

        if satisfied_predicates != satisfied_predicates_cached:
            vr_cs.refresh_condition(switch=False)
            satisfied_predicates_cached = satisfied_predicates

        if igtn_task.simulator.query_vr_event('right_controller', 'overlay_toggle'):
            vr_cs.refresh_condition()

        if igtn_task.simulator.query_vr_event('left_controller', 'overlay_toggle'):
            vr_cs.toggle_show_state()

        if not args.disable_save:
            vr_writer.process_frame(s)

    s.disconnect()


if __name__ == "__main__":
    main()
