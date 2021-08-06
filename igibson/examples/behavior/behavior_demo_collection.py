"""
Main BEHAVIOR demo collection entrypoint
"""

import argparse
import copy
import datetime
import os

import bddl
import numpy as np

import igibson
from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrConditionSwitcher, VrSettings
from igibson.simulator import Simulator
from igibson.utils.ig_logging import IGLogWriter

POST_TASK_STEPS = 200
PHYSICS_WARMING_STEPS = 200


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
    parser = argparse.ArgumentParser(description="Run and collect an ATUS demo")
    parser.add_argument(
        "--task", type=str, required=True, nargs="?", help="Name of ATUS activity matching parent folder in bddl."
    )
    parser.add_argument(
        "--task_id",
        type=int,
        required=True,
        choices=task_id_choices,
        nargs="?",
        help="BDDL integer ID, matching suffix of bddl.",
    )
    parser.add_argument("--vr_log_path", type=str, help="Path (and filename) of vr log")
    parser.add_argument(
        "--scene", type=str, choices=scene_choices, nargs="?", help="Scene name/ID matching iGibson interactive scenes."
    )
    parser.add_argument("--disable_save", action="store_true", help="Whether to disable saving logfiles.")
    parser.add_argument(
        "--disable_scene_cache", action="store_true", help="Whether to disable using pre-initialized scene caches."
    )
    parser.add_argument("--profile", action="store_true", help="Whether to print profiling data.")
    parser.add_argument(
        "--no_vr", action="store_true", help="Whether to turn off VR recording and save random actions."
    )
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum number of steps to record before stopping.")
    return parser.parse_args()


def main():
    args = parse_args()
    bddl.set_backend("iGibson")
    collect_demo(
        args.task,
        args.task_id,
        args.scene,
        args.vr_log_path,
        args.disable_save,
        args.max_steps,
        args.no_vr,
        args.disable_scene_cache,
        args.profile,
    )


def collect_demo(
    task,
    task_id,
    scene,
    vr_log_path=None,
    disable_save=False,
    max_steps=-1,
    no_vr=False,
    disable_scene_cache=False,
    profile=False,
):
    # HDR files for PBR rendering
    hdr_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_02.hdr")
    hdr_texture2 = os.path.join(igibson.ig_dataset_path, "scenes", "background", "probe_03.hdr")
    light_modulation_map_filename = os.path.join(
        igibson.ig_dataset_path, "scenes", "Rs_int", "layout", "floor_lighttype_0.png"
    )
    background_texture = os.path.join(igibson.ig_dataset_path, "scenes", "background", "urban_street_01.jpg")

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
        light_dimming_factor=1.0,
    )

    # VR system settings
    mode = "headless" if no_vr else "vr"
    s = Simulator(
        mode=mode,
        rendering_settings=vr_rendering_settings,
        vr_settings=VrSettings(use_vr=True),
        physics_timestep=1 / 300.0,
        render_timestep=1 / 30.0,
    )
    igbhvr_act_inst = iGBEHAVIORActivityInstance(task, task_id)

    scene_kwargs = None
    online_sampling = True

    if not disable_scene_cache:
        scene_kwargs = {
            "urdf_file": "{}_task_{}_{}_0_fixed_furniture".format(scene, task, task_id),
        }
        online_sampling = False

    igbhvr_act_inst.initialize_simulator(
        simulator=s, scene_id=scene, scene_kwargs=scene_kwargs, load_clutter=True, online_sampling=online_sampling
    )
    vr_agent = igbhvr_act_inst.simulator.robots[0]

    if not no_vr:
        vr_cs = VrConditionSwitcher(
            igbhvr_act_inst.simulator, igbhvr_act_inst.show_instruction, igbhvr_act_inst.iterate_instruction
        )

    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if vr_log_path is None:
            vr_log_path = "{}_{}_{}_{}.hdf5".format(task, task_id, scene, timestamp)
        log_writer = IGLogWriter(
            s,
            log_filepath=vr_log_path,
            task=igbhvr_act_inst,
            store_vr=False if no_vr else True,
            vr_robot=vr_agent,
            profiling_mode=profile,
            filter_objects=True,
        )
        log_writer.set_up_data_storage()

    satisfied_predicates_cached = {}
    post_task_steps = copy.deepcopy(POST_TASK_STEPS)
    physics_warming_steps = copy.deepcopy(PHYSICS_WARMING_STEPS)

    steps = 0
    while max_steps < 0 or steps < max_steps:
        igbhvr_act_inst.simulator.step(print_stats=profile)
        task_done, satisfied_predicates = igbhvr_act_inst.check_success()

        if no_vr:
            if steps < 2:
                action = np.zeros((28,))
                action[19] = 1
                action[27] = 1
            else:
                action = np.random.uniform(-0.01, 0.01, size=(28,))
        else:
            action = igbhvr_act_inst.simulator.gen_vr_robot_action()
            if steps < physics_warming_steps:
                action = np.zeros_like(action)

        vr_agent.update(action)

        if not no_vr:
            if satisfied_predicates != satisfied_predicates_cached:
                vr_cs.refresh_condition(switch=False)
                satisfied_predicates_cached = satisfied_predicates

            if igbhvr_act_inst.simulator.query_vr_event("right_controller", "overlay_toggle"):
                vr_cs.refresh_condition()

            if igbhvr_act_inst.simulator.query_vr_event("left_controller", "overlay_toggle"):
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
