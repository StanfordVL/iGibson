"""
BEHAVIOR RL episodes replay entrypoint
"""

import argparse
import datetime
import os
import pprint

import bddl
import h5py
import numpy as np

import igibson
from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.simulator import Simulator
from igibson.utils.git_utils import project_git_info
from igibson.utils.ig_logging import IGLogReader, IGLogWriter
from igibson.utils.utils import parse_str_config


def verify_determinism(in_log_path, out_log_path):
    is_deterministic = True
    with h5py.File(in_log_path) as original_file, h5py.File(out_log_path) as new_file:
        for obj in original_file["physics_data"]:
            for attribute in original_file["physics_data"][obj]:
                is_close = np.isclose(
                    original_file["physics_data"][obj][attribute], new_file["physics_data"][obj][attribute]
                ).all()
                is_deterministic = is_deterministic and is_close
                if not is_close:
                    print("Mismatch for obj {} with mismatched attribute {}".format(obj, attribute))
    return bool(is_deterministic)


def parse_args():
    parser = argparse.ArgumentParser(description="Run and collect an ATUS demo")
    parser.add_argument("--vr_log_path", type=str, help="Path (and filename) of vr log to replay")
    parser.add_argument(
        "--vr_replay_log_path", type=str, help="Path (and filename) of file to save replay to (for debugging)"
    )
    parser.add_argument(
        "--frame_save_path",
        type=str,
        help="Path to save frames (frame number added automatically, as well as .jpg extension)",
    )
    parser.add_argument(
        "--disable_save",
        action="store_true",
        help="Whether to disable saving log of replayed trajectory, used for validation.",
    )
    parser.add_argument("--profile", action="store_true", help="Whether to print profiling data.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["headless", "vr", "simple"],
        help="Whether to disable replay through VR and use iggui instead.",
    )
    return parser.parse_args()


def replay_demo(
    in_log_path,
    out_log_path=None,
    disable_save=False,
    frame_save_path=None,
    verbose=True,
    mode="headless",
    start_callbacks=[],
    step_callbacks=[],
    end_callbacks=[],
    profile=False,
):
    """
    Replay a BEHAVIOR demo.

    Note that this returns, but does not check for determinism. Use safe_replay_demo to assert for determinism
    when using in scenarios where determinism is important.

    @param in_log_path: the path of the BEHAVIOR demo log to replay.
    @param out_log_path: the path of the new BEHAVIOR demo log to save from the replay.
    @param frame_save_path: the path to save frame images to. None to disable frame image saving.
    @param mode: which rendering mode ("headless", "simple", "vr"). In simple mode, the demo will be replayed with simple robot view.
    @param disable_save: Whether saving the replay as a BEHAVIOR demo log should be disabled.
    @param profile: Whether the replay should be profiled, with profiler output to stdout.
    @param start_callback: A callback function that will be called immediately before starting to replay steps. Should
        take a single argument, an iGBEHAVIORActivityInstance.
    @param step_callback: A callback function that will be called immediately following each replayed step. Should
        take a single argument, an iGBEHAVIORActivityInstance.
    @param end_callback: A callback function that will be called when replay has finished. Should take a single
        argument, an iGBEHAVIORActivityInstance.
    @return if disable_save is True, returns None. Otherwise, returns a boolean indicating if replay was deterministic.
    """
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

    # Check mode
    assert mode in ["headless", "vr", "simple"]

    # Initialize settings to save action replay frames
    vr_settings = VrSettings(config_str=IGLogReader.read_metadata_attr(in_log_path, "/metadata/vr_settings"))
    vr_settings.set_frame_save_path(frame_save_path)

    task = IGLogReader.read_metadata_attr(in_log_path, "/metadata/atus_activity")
    task_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/activity_definition")
    scene = IGLogReader.read_metadata_attr(in_log_path, "/metadata/scene_id")
    physics_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/physics_timestep")
    render_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/render_timestep")
    filter_objects = IGLogReader.read_metadata_attr(in_log_path, "/metadata/filter_objects")

    logged_git_info = IGLogReader.read_metadata_attr(in_log_path, "/metadata/git_info")
    logged_git_info = parse_str_config(logged_git_info)
    git_info = project_git_info()
    pp = pprint.PrettyPrinter(indent=4)

    for key in logged_git_info.keys():
        logged_git_info[key].pop("directory", None)
        git_info[key].pop("directory", None)
        if logged_git_info[key] != git_info[key] and verbose:
            print("Warning, difference in git commits for repo: {}. This may impact deterministic replay".format(key))
            print("Logged git info:\n")
            pp.pprint(logged_git_info[key])
            print("Current git info:\n")
            pp.pprint(git_info[key])

    # VR system settings
    s = Simulator(
        mode=mode,
        physics_timestep=physics_timestep,
        render_timestep=render_timestep,
        rendering_settings=vr_rendering_settings,
        vr_settings=vr_settings,
        image_width=1280,
        image_height=720,
    )

    igtn_task = iGBEHAVIORActivityInstance(task, task_id)
    igtn_task.initialize_simulator(
        simulator=s,
        scene_id=scene,
        scene_kwargs={
            "urdf_file": "{}_neurips_task_{}_{}_0_fixed_furniture".format(scene, task, task_id),
        },
        load_clutter=True,
        online_sampling=False,
    )
    vr_agent = igtn_task.simulator.robots[0]
    vr_agent.activate()
    igtn_task.reset_scene(snapshot_id=igtn_task.initial_state)
    # set the constraints to the current poses
    vr_agent.update(np.zeros(28))

    if not in_log_path:
        raise RuntimeError("Must provide a VR log path to run action replay!")
    log_reader = IGLogReader(in_log_path, log_status=False)

    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if out_log_path == None:
            out_log_path = "{}_{}_{}_{}_replay.hdf5".format(task, task_id, scene, timestamp)

        log_writer = IGLogWriter(
            s,
            log_filepath=out_log_path,
            task=igtn_task,
            store_vr=False,
            vr_robot=vr_agent,
            profiling_mode=profile,
            filter_objects=filter_objects,
        )
        log_writer.set_up_data_storage()

    for callback in start_callbacks:
        callback(igtn_task, log_reader)

    task_done = False
    while log_reader.get_data_left_to_read():

        action = log_reader.get_agent_action("vr_robot")
        # Get relevant VR action data and update VR agent
        vr_agent.update(action)

        if not disable_save:
            log_writer.process_frame()

        igtn_task.simulator.step(print_stats=profile)
        task_done, _ = igtn_task.check_success()

        # Set camera each frame
        if mode == "vr":
            log_reader.set_replay_camera(s)

        for callback in step_callbacks:
            callback(igtn_task, log_reader)

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

    demo_statistics = {}
    for callback in end_callbacks:
        callback(igtn_task, log_reader)

    s.disconnect()

    is_deterministic = None
    if not disable_save:
        log_writer.end_log_session()
        is_deterministic = verify_determinism(in_log_path, out_log_path)
        print("Demo was deterministic: ", is_deterministic)

    demo_statistics = {
        "deterministic": is_deterministic,
        "task": task,
        "task_id": int(task_id),
        "scene": scene,
        "task_done": task_done,
        "total_frame_num": log_reader.total_frame_num,
    }
    return demo_statistics


def safe_replay_demo(*args, **kwargs):
    """Replays a demo, asserting that it was deterministic."""
    demo_statistics = replay_demo(*args, **kwargs)
    assert (
        demo_statistics["deterministic"] == True
    ), "Replay was not deterministic (or was executed with disable_save=True)."


def main():
    args = parse_args()
    bddl.set_backend("iGibson")
    replay_demo(
        args.vr_log_path,
        out_log_path=args.vr_replay_log_path,
        disable_save=args.disable_save,
        frame_save_path=args.frame_save_path,
        mode=args.mode,
        profile=args.profile,
    )


if __name__ == "__main__":
    main()
