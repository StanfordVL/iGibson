"""
  Script to convert BEHAVIOR virtual reality demos to dataset compatible with imitation learning
"""

import argparse
import json
import logging
import os
import pprint
import types
from pathlib import Path

import bddl
import h5py
import numpy as np
import pandas as pd

import igibson
from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.sensors.vision_sensor import VisionSensor
from igibson.simulator import Simulator
from igibson.utils.git_utils import project_git_info
from igibson.utils.ig_logging import IGLogReader
from igibson.utils.utils import parse_str_config


def replay_imitation_demo(
    in_log_path,
    frame_save_path=None,
    verbose=True,
    mode="headless",
    profile=False,
):
    """
    Replay a BEHAVIOR demo.

    Note that this returns, but does not check for determinism. Use safe_replay_demo to assert for determinism
    when using in scenarios where determinism is important.

    @param in_log_path: the path of the BEHAVIOR demo log to replay.
    @param frame_save_path: the path to save frame images to. None to disable frame image saving.
    @param mode: which rendering mode ("headless", "headless_tensor", "gui_non_interactive", "vr"). In gui_non_interactive
        mode, the demo will be replayed with simple robot view.
    @param disable_save: Whether saving the replay as a BEHAVIOR demo log should be disabled.
    @param profile: Whether the replay should be profiled, with profiler output to stdout.
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
    assert mode in ["headless", "headless_tensor", "vr", "gui_non_interactive"]

    # Initialize settings to save action replay frames
    vr_settings = VrSettings(config_str=IGLogReader.read_metadata_attr(in_log_path, "/metadata/vr_settings"))
    vr_settings.set_frame_save_path(frame_save_path)

    activity = IGLogReader.read_metadata_attr(in_log_path, "/metadata/atus_activity")
    activity_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/activity_definition")
    scene = IGLogReader.read_metadata_attr(in_log_path, "/metadata/scene_id")
    physics_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/physics_timestep")
    render_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/render_timestep")
    instance_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/instance_id")
    urdf_file = IGLogReader.read_metadata_attr(in_log_path, "/metadata/urdf_file")

    if urdf_file is None:
        urdf_file = "{}_task_{}_{}_0_fixed_furniture".format(scene, activity, activity_id)

    if instance_id is None:
        instance_id = 0

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
            "urdf_file": urdf_file,
        },
        load_clutter=True,
        online_sampling=False,
    )
    vr_agent = igbhvr_act_inst.simulator.robots[0]
    log_reader = IGLogReader(in_log_path, log_status=False)

    env = types.SimpleNamespace()
    env.simulator = igbhvr_act_inst.simulator
    env.robots = igbhvr_act_inst.simulator.robots
    env.config = {}
    vision_sensor = VisionSensor(env, ["rgb", "highlight", "depth", "seg", "ins_seg"])

    episode_identifier = "_".join(os.path.splitext(in_log_path)[0].split("_")[-2:])
    episode_out_log_path = "processed_hdf5s/{}_{}_{}_{}_{}_episode.hdf5".format(
        activity, activity_id, scene, instance_id, episode_identifier
    )
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
        state["action"] = log_reader.read_value("agent_actions/vr_robot")

        igbhvr_act_inst.simulator.step(print_stats=profile)
        task_done |= igbhvr_act_inst.check_success()[0]

        # Set camera each frame
        if mode == "vr":
            log_reader.set_replay_camera(s)

        # Get relevant VR action data and update VR agent
        vr_agent.update(log_reader.get_agent_action("vr_robot"))

        for key, value in state.items():
            if key not in state_history:
                state_history[key] = []
            state_history[key].append(value)

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

    s.disconnect()

    demo_statistics = {
        "task": activity,
        "task_id": int(activity_id),
        "scene": scene,
        "task_done": task_done,
        "total_frame_num": log_reader.total_frame_num,
    }
    hf.close()
    return demo_statistics


def generate_imitation_dataset(demo_root, log_manifest, out_dir, skip_existing=True, save_frames=False):
    """
    Execute imitation dataset generation on a batch of BEHAVIOR demos.

    @param demo_root: Directory containing the demo files listed in the manifests.
    @param log_manifest: The manifest file containing list of BEHAVIOR demos to batch over.
    @param out_dir: Directory to store results in.
    @param skip_existing: Whether demos with existing output logs should be skipped.
    @param save_frames: Whether the demo's frames should be saved alongside statistics.
    """
    logger = logging.getLogger()
    logger.disabled = True

    bddl.set_backend("iGibson")

    demo_list = pd.read_csv(log_manifest)

    for idx, demo in enumerate(demo_list["demos"]):
        if "replay" in demo:
            continue

        demo_name = os.path.splitext(demo)[0]
        demo_path = os.path.join(demo_root, demo)
        log_path = os.path.join(out_dir, demo_name + ".json")

        if skip_existing and os.path.exists(log_path):
            print("Skipping existing demo: {}, {} out of {}".format(demo, idx, len(demo_list["demos"])))
            continue

        print("Replaying demo: {}, {} out of {}".format(demo, idx, len(demo_list["demos"])))

        curr_frame_save_path = None
        if save_frames:
            curr_frame_save_path = os.path.join(out_dir, demo_name + ".mp4")

        try:
            demo_information = replay_imitation_demo(
                in_log_path=demo_path,
                frame_save_path=curr_frame_save_path,
                mode="headless",
                verbose=False,
            )
            demo_information["failed"] = False
            demo_information["filename"] = Path(demo).name

        except Exception as e:
            print("Demo failed withe error: ", e)
            demo_information = {"demo_id": Path(demo).name, "failed": True, "failure_reason": str(e)}

        with open(log_path, "w") as file:
            json.dump(demo_information, file)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect metrics from BEHAVIOR demos in manifest.")
    parser.add_argument("--demo_root", type=str, help="Directory containing demos listed in the manifest.")
    parser.add_argument("--log_manifest", type=str, help="Plain text file consisting of list of demos to replay.")
    parser.add_argument("--out_dir", type=str, help="Directory to store results in.")
    return parser.parse_args()


def main():
    args = parse_args()
    generate_imitation_dataset(args.demo_root, args.log_manifest, args.out_dir)


if __name__ == "__main__":
    main()
