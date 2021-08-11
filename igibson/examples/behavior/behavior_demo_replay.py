"""
Main BEHAVIOR demo replay entrypoint
"""

import argparse
import datetime
import os
import pprint

import types
from igibson.sensors.vision_sensor import VisionSensor
import bddl
import h5py
import numpy as np
import pybullet as p
import copy

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

    activity = IGLogReader.read_metadata_attr(in_log_path, "/metadata/atus_activity")
    activity_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/activity_definition")
    scene = IGLogReader.read_metadata_attr(in_log_path, "/metadata/scene_id")
    physics_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/physics_timestep")
    render_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/render_timestep")
    filter_objects = IGLogReader.read_metadata_attr(in_log_path, "/metadata/filter_objects")

    logged_git_info = IGLogReader.read_metadata_attr(in_log_path, "/metadata/git_info")
    logged_git_info = parse_str_config(logged_git_info)
    git_info = project_git_info()
    pp = pprint.PrettyPrinter(indent=4)

    for key in logged_git_info.keys():
        if key not in git_info:
            print(
                "Warning: {} not present in current git info. It might be installed through PyPI, "
                "so its version cannot be validated.".format(key)
            )
            continue

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
        image_width=128,
        image_height=128,
    )

    igbhvr_act_inst = iGBEHAVIORActivityInstance(activity, activity_id)
    igbhvr_act_inst.initialize_simulator(
        simulator=s,
        scene_id=scene,
        scene_kwargs={
            "urdf_file": "{}_task_{}_{}_0_fixed_furniture".format(scene, activity, activity_id),
        },
        load_clutter=True,
        online_sampling=False,
    )
    vr_agent = igbhvr_act_inst.simulator.robots[0]
    log_reader = IGLogReader(in_log_path, log_status=False)

    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if out_log_path == None:
            out_log_path = "{}_{}_{}_{}_replay.hdf5".format(activity, activity_id, scene, timestamp)

        log_writer = IGLogWriter(
            s,
            log_filepath=out_log_path,
            task=igbhvr_act_inst,
            store_vr=False,
            vr_robot=vr_agent,
            profiling_mode=profile,
            filter_objects=filter_objects,
        )
        log_writer.set_up_data_storage()

    for callback in start_callbacks:
        callback(igbhvr_act_inst, log_reader)

    env = types.SimpleNamespace()
    env.simulator = igbhvr_act_inst.simulator
    env.robots = igbhvr_act_inst.simulator.robots
    env.config = {}
    vision_sensor = VisionSensor(env, ['rgb', 'highlight', 'depth', 'seg', 'ins_seg'])

        
    episode_identifier = "_".join(os.path.splitext(in_log_path)[0].split("_")[-2:])
    episode_out_log_path = "processed_hdf5s/{}_{}_{}_{}_episode.hdf5".format(activity, activity_id, scene, episode_identifier)
    hf = h5py.File(episode_out_log_path, "w")
    # hf.attrs["/metadata/collection_date"] = IGLogReader.read_metadata_attr(in_log_path, "/metadata/start_time")
    hf.attrs["/metadata/physics_timestep"] = physics_timestep
    hf.attrs["/metadata/render_timestep"] = render_timestep
    hf.attrs["/metadata/activity"] = activity
    hf.attrs["/metadata/activity_id"] = activity_id
    hf.attrs["/metadata/scene_id"] = scene
    hf.attrs["/metadata/vr_settings"] = igbhvr_act_inst.simulator.vr_settings.dump_vr_settings()

    state_history = {}
    task_done = False

    for _, obj in igbhvr_act_inst.object_scope.items():
        if obj.category in ["agent", "room_floor"]:
            continue
        obj.highlight()

    while log_reader.get_data_left_to_read():
        state = {}
        state.update(vision_sensor.get_obs(env))
        state["task_obs"] = igbhvr_act_inst.get_task_obs(env)
        state["proprioception"] = np.array(env.robots[0].get_proprioception())
        state["action"] = log_reader.read_value('agent_actions/vr_robot')
        if np.max(state['seg']) > 400: 
            import pdb; pdb.set_trace()

        igbhvr_act_inst.simulator.step(print_stats=profile)
        task_done, _ = igbhvr_act_inst.check_success()

        # Set camera each frame
        if mode == "vr":
            log_reader.set_replay_camera(s)

        for callback in step_callbacks:
            callback(igbhvr_act_inst, log_reader)

        # Get relevant VR action data and update VR agent
        vr_agent.update(log_reader.get_agent_action("vr_robot"))

        for key, value in state.items():
            if key not in state_history:
                state_history[key] = []
            state_history[key].append(value)
        

        if not disable_save:
            log_writer.process_frame()

    print("Demo was succesfully completed: ", task_done)

    print("Compressing demo data")
    for key, value in state_history.items():
        if key == "action":
            dtype = np.float64
        else:
            dtype = np.float32
        hf.create_dataset(key, data=np.stack(value), dtype=dtype, compression="lzf")
    print("Compression complete")

    demo_statistics = {}
    for callback in end_callbacks:
        callback(igbhvr_act_inst, log_reader)

    s.disconnect()

    is_deterministic = None
    if not disable_save:
        log_writer.end_log_session()
        is_deterministic = verify_determinism(in_log_path, out_log_path)
        print("Demo was deterministic: ", is_deterministic)

    demo_statistics = {
        "deterministic": is_deterministic,
        "task": activity,
        "task_id": int(activity_id),
        "scene": scene,
        "task_done": task_done,
        "total_frame_num": log_reader.total_frame_num,
    }
    hf.close()
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
