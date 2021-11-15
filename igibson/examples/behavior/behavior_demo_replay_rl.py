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
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.simulator import Simulator
from igibson.utils.git_utils import project_git_info
from igibson.utils.ig_logging import IGLogReader, IGLogWriter
from igibson.utils.utils import parse_config, parse_str_config


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
    parser.add_argument("--vr_log_path", type=str, help="Path (and filename) of vr log to replay", required=True)
    parser.add_argument(
        "--vr_replay_log_path",
        type=str,
        help="Path (and filename) of file to save replay to (for debugging)",
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
    parser.add_argument(
        "--config", help="which config file to use [default: use yaml files in examples/configs]", required=True
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["headless", "headless_tensor", "vr", "gui_non_interactive"],
        help="Mode to run simulator in",
        required=True,
    )
    return parser.parse_args()


def replay_demo(
    in_log_path,
    config_file,
    mode,
    out_log_path=None,
    disable_save=False,
):
    """
    Replay a BEHAVIOR demo.

    Note that this returns, but does not check for determinism. Use safe_replay_demo to assert for determinism
    when using in scenarios where determinism is important.

    @param in_log_path: the path of the BEHAVIOR demo log to replay.
    @param config_file: environment config file
    @param mode: simulator mode
    @param out_log_path: the path of the new BEHAVIOR demo log to save from the replay.
    @param mode: which rendering mode ("headless", "headless_tensor", "gui_non_interactive", "vr"). In gui_non_interactive
        mode, the demo will be replayed with simple robot view.
    @param disable_save: Whether saving the replay as a BEHAVIOR demo log should be disabled.
    @return if disable_save is True, returns None. Otherwise, returns a boolean indicating if replay was deterministic.
    """

    task = IGLogReader.read_metadata_attr(in_log_path, "/metadata/atus_activity")
    task_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/activity_definition")
    scene_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/scene_id")
    physics_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/physics_timestep")
    render_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/render_timestep")
    filter_objects = IGLogReader.read_metadata_attr(in_log_path, "/metadata/filter_objects")
    instance_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/instance_id")
    urdf_file = IGLogReader.read_metadata_attr(in_log_path, "/metadata/urdf_file")

    config = parse_config(config_file)
    config["task"] = task
    config["task_id"] = task_id
    config["scene_id"] = scene_id
    config["instance_id"] = instance_id
    config["urdf_file"] = urdf_file

    env = iGibsonEnv(config_file=config, mode=mode, action_timestep=render_timestep, physics_timestep=physics_timestep)
    env.reset()
    vr_agent = env.robots[0]

    log_reader = IGLogReader(in_log_path, log_status=False)

    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if out_log_path == None:
            out_log_path = "{}_{}_{}_{}_replay.hdf5".format(task, task_id, scene_id, timestamp)

        log_writer = IGLogWriter(
            env.simulator,
            log_filepath=out_log_path,
            task=env.task,
            store_vr=False,
            vr_robot=vr_agent,
            profiling_mode=False,
            filter_objects=filter_objects,
        )
        log_writer.set_up_data_storage()

    task_done = False
    while log_reader.get_data_left_to_read():

        action = log_reader.get_agent_action("vr_robot")
        # Get relevant VR action data and update VR agent
        vr_agent.apply_action(action)

        env.simulator.step()
        task_done |= env.task.check_success()[0]

        if not disable_save:
            log_writer.process_frame()

        # Per-step determinism check. Activate if necessary.
        things_to_compare = [thing for thing in log_writer.name_path_data if thing[0] == "physics_data"]
        for thing in things_to_compare:
            thing_path = "/".join(thing)
            fc = log_reader.frame_counter % log_writer.frames_before_write
            if fc == log_writer.frames_before_write - 1:
                continue
            replayed = log_writer.get_data_for_name_path(thing)[fc]
            original = log_reader.read_value(thing_path)
            if not np.all(replayed == original):
                print("%s not equal in %d" % (thing_path, log_reader.frame_counter))
                assert False
            if not np.isclose(replayed, original).all():
                print("%s not close in %d" % (thing_path, log_reader.frame_counter))
                assert False

    print("Demo was succesfully completed: ", task_done)

    env.close()

    is_deterministic = None
    if not disable_save:
        log_writer.end_log_session()
        is_deterministic = verify_determinism(in_log_path, out_log_path)
        print("Demo was deterministic: ", is_deterministic)

    demo_statistics = {
        "deterministic": is_deterministic,
        "task": task,
        "task_id": int(task_id),
        "scene": scene_id,
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
    replay_demo(
        in_log_path=args.vr_log_path,
        config_file=args.config,
        mode=args.mode,
        out_log_path=args.vr_replay_log_path,
        disable_save=args.disable_save,
    )


if __name__ == "__main__":
    main()
