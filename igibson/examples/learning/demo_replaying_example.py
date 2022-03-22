import argparse
import datetime
import logging
import os
import pprint
import tempfile

import h5py
import numpy as np

import igibson
from igibson.envs.igibson_env import iGibsonEnv
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.utils.git_utils import project_git_info
from igibson.utils.ig_logging import IGLogReader, IGLogWriter
from igibson.utils.utils import parse_config, parse_str_config


def verify_determinism(in_log_path, out_log_path):
    is_deterministic = True
    with h5py.File(in_log_path) as original_file, h5py.File(out_log_path) as new_file:
        if sorted(list(original_file["physics_data"])) != sorted(list(new_file["physics_data"])):
            logging.warning(
                "Object in the original demo and the replay have different number of objects logged: {} vs. {}".format(
                    sorted(list(original_file["physics_data"])), sorted(list(new_file["physics_data"]))
                )
            )
            is_deterministic = False
        else:
            for obj in original_file["physics_data"]:
                for attribute in original_file["physics_data"][obj]:
                    is_close = np.isclose(
                        original_file["physics_data"][obj][attribute], new_file["physics_data"][obj][attribute]
                    )
                    is_deterministic = is_deterministic and is_close.all()
                    if not is_close.all():
                        logging.warning(
                            "Mismatch for obj {} with mismatched attribute {} starting at timestep {}".format(
                                obj, attribute, np.where(is_close == False)[0][0]
                            )
                        )
    return bool(is_deterministic)


def parse_args(defaults=False):

    args_dict = dict()
    args_dict["in_demo_file"] = os.path.join(
        igibson.ig_dataset_path,
        "tests",
        "cleaning_windows_example.hdf5",
    )
    args_dict["config_file"] = os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml")

    if not defaults:
        parser = argparse.ArgumentParser(description="Replay a BEHAVIOR demo")
        parser.add_argument("--in_demo_file", type=str, help="Path (and filename) of demo to replay")
        parser.add_argument(
            "--replay_demo_file",
            type=str,
            help="Path (and filename) of file to save replay to (for debugging)",
        )
        parser.add_argument(
            "--frame_save_dir",
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
            choices=["headless", "headless_tensor", "vr", "gui_non_interactive"],
            help="Mode for replaying",
            default="headless",
        )
        parser.add_argument(
            "--config_file",
            help="which config file to use [default: use yaml files in examples/configs]",
            default=args_dict["config_file"],
        )
        args = parser.parse_args()

        args_dict["in_demo_file"] = args.in_demo_file
        args_dict["replay_demo_file"] = args.replay_demo_file
        args_dict["disable_save"] = args.disable_save
        args_dict["frame_save_dir"] = args.frame_save_dir
        args_dict["mode"] = args.mode
        args_dict["profile"] = args.profile
        args_dict["config_file"] = args.config_file

    return args_dict


def replay_demo(
    in_demo_file,
    replay_demo_file=None,
    disable_save=False,
    frame_save_dir=None,
    verbose=True,
    mode="headless",
    config_file=os.path.join(igibson.configs_path, "behavior_robot_vr_behavior_task.yaml"),
    start_callbacks=[],
    step_callbacks=[],
    end_callbacks=[],
    profile=False,
    image_size=(1280, 720),
    use_pb_gui=False,
):
    """
    Replay a demo of a task.

    Note that this returns, but does not check for determinism. Use safe_replay_demo to assert for determinism
    when using in scenarios where determinism is important.

    @param in_demo_file: the path and filename of the BEHAVIOR demo to replay.
    @param replay_demo_file: the path and filename of the new BEHAVIOR demo to save from the replay.
    @param disable_save: Whether saving the replay as a BEHAVIOR demo log should be disabled.
    @param frame_save_dir: the path to save frame images to. None to disable frame image saving.
    @param verbose: Whether to print out git diff in detail
    @param mode: which rendering mode ("headless", "headless_tensor", "gui_non_interactive", "vr"). In gui_non_interactive
        mode, the demo will be replayed with simple robot view.
    @param config_file: environment config file
    @param disable_save: Whether saving the replay as a BEHAVIOR demo log should be disabled.
    @param profile: Whether the replay should be profiled, with profiler output to stdout.
    @param start_callback: A callback function that will be called immediately before starting to replay steps. Should
        take two arguments: iGibsonEnv and IGLogReader
    @param step_callback: A callback function that will be called immediately following each replayed step. Should
        take two arguments: iGibsonEnv and IGLogReader
    @param end_callback: A callback function that will be called when replay has finished. Should
        take two arguments: iGibsonEnv and IGLogReader
    @param profile: Whether the replay should be profiled, with profiler output to stdout.
    @param image_size: The image size that should be used by the renderer.
    @param use_pb_gui: display the interactive pybullet gui (for debugging)
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
    rendering_setting = MeshRendererSettings(
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
    vr_settings = VrSettings(config_str=IGLogReader.read_metadata_attr(in_demo_file, "/metadata/vr_settings"))
    vr_settings.set_frame_save_path(frame_save_dir)

    # Get the information from the input log file
    task = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/atus_activity")
    task_id = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/activity_definition")
    scene = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/scene_id")
    physics_timestep = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/physics_timestep")
    render_timestep = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/render_timestep")
    filter_objects = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/filter_objects")
    instance_id = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/instance_id")
    urdf_file = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/urdf_file")

    if urdf_file is None:
        urdf_file = "{}_task_{}_{}_0_fixed_furniture".format(scene, task, task_id)

    if instance_id is None:
        instance_id = 0

    logged_git_info = IGLogReader.read_metadata_attr(in_demo_file, "/metadata/git_info")
    logged_git_info = parse_str_config(logged_git_info)

    # Get current git info
    git_info = project_git_info()
    pp = pprint.PrettyPrinter(indent=4)

    # Check if the current git info and the one in the log are the same
    for key in logged_git_info.keys():
        if key not in git_info:
            logging.info(
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

    # Get some information from the config and some other copy it from the input log
    config = parse_config(config_file)
    config["task"] = task
    config["task_id"] = task_id
    config["scene_id"] = scene
    config["instance_id"] = instance_id
    config["urdf_file"] = urdf_file
    config["image_width"] = image_size[0]
    config["image_height"] = image_size[1]
    config["online_sampling"] = False

    print("Creating environment and resetting it")
    env = iGibsonEnv(
        config_file=config,
        mode=mode,
        action_timestep=render_timestep,
        physics_timestep=physics_timestep,
        rendering_settings=rendering_setting,
        vr_settings=vr_settings,
        use_pb_gui=use_pb_gui,
    )
    env.reset()
    robot = env.robots[0]

    if not in_demo_file:
        raise RuntimeError("Must provide a log path to run action replay!")
    log_reader = IGLogReader(in_demo_file, log_status=False)

    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if replay_demo_file is None:
            temp_folder = tempfile.TemporaryDirectory()
            replay_demo_file = "{}_{}_{}_{}_{}_replay.hdf5".format(task, task_id, scene, instance_id, timestamp)
            replay_demo_file = os.path.join(temp_folder.name, replay_demo_file)

        log_writer = IGLogWriter(
            env.simulator,
            log_filepath=replay_demo_file,
            task=env.task,
            store_vr=False,
            vr_robot=robot,
            profiling_mode=profile,
            filter_objects=filter_objects,
        )
        log_writer.set_up_data_storage()

    try:
        for callback in start_callbacks:
            callback(env, log_reader)

        task_done = False

        print("Replaying demo")
        while log_reader.get_data_left_to_read():
            env.step(log_reader.get_agent_action("vr_robot"))
            task_done |= env.task.check_success()[0]

            # Set camera each frame
            if mode == "vr":
                log_reader.set_replay_camera(env.simulator)

            for callback in step_callbacks:
                callback(env, log_reader)

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

        print("Demo ended in success: {}".format(task_done))

        for callback in end_callbacks:
            callback(env, log_reader)
    finally:
        print("End of the replay.")
        if not disable_save:
            log_writer.end_log_session()
        env.close()

    is_deterministic = None
    if not disable_save:
        is_deterministic = verify_determinism(in_demo_file, replay_demo_file)
        print("Demo was deterministic: {}".format(is_deterministic))

    demo_statistics = {
        "deterministic": is_deterministic,
        "task": task,
        "task_id": int(task_id),
        "scene": scene,
        "task_done": task_done,
        "total_frame_num": log_reader.total_frame_num,
    }
    return demo_statistics


def replay_demo_with_determinism_check(*args, **kwargs):
    """Replays a demo, asserting that it was deterministic."""
    demo_statistics = replay_demo(*args, **kwargs)
    assert (
        demo_statistics["deterministic"] == True
    ), "Replay was not deterministic (or was executed with disable_save=True)."


def main(selection="user", headless=False, short_exec=False):
    """
    Example of how to replay a previously recorded demo of a task
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Assuming that if selection!="user", headless=True, short_exec=True, we are calling it from tests and we
    # do not want to parse args (it would fail because the calling function is pytest "testfile.py")
    if not (selection != "user" and headless and short_exec):
        args_dict = parse_args()
    else:
        args_dict = parse_args(True)

    in_demo_file = args_dict.pop("in_demo_file")
    replay_demo(in_demo_file, **args_dict)


RUN_AS_TEST = False  # Change to True to run this example in test mode
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    if RUN_AS_TEST:
        main(selection="random", headless=True, short_exec=True)
    else:
        main()
