"""
Debugging script to bypass taksnet
"""

import argparse
import datetime
import os

import igibson
from igibson.metrics.agent import AgentMetric
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.ig_logging import IGLogReader, IGLogWriter

POST_TASK_STEPS = 200
PHYSICS_WARMING_STEPS = 200


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

    physics_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/physics_timestep")
    render_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/render_timestep")
    filter_objects = IGLogReader.read_metadata_attr(in_log_path, "/metadata/filter_objects")

    scene_id = "Rs_int"
    # VR system settings
    s = Simulator(mode="vr", rendering_settings=vr_rendering_settings, vr_settings=VrSettings(use_vr=True))

    scene = InteractiveIndoorScene(
        scene_id, load_object_categories=["walls", "floors", "ceilings"], load_room_types=["kitchen"]
    )

    vr_agent = BehaviorRobot(s)
    s.import_ig_scene(scene)
    s.import_behavior_robot(vr_agent)
    s.register_main_vr_robot(vr_agent)
    vr_agent.set_position_orientation([0, 0, 1.5], [0, 0, 0, 1])

    log_reader = IGLogReader(in_log_path, log_status=False)

    log_writer = None
    if not disable_save:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        if out_log_path == None:
            out_log_path = "{}_{}_replay.hdf5".format(scene_id, timestamp)

        log_writer = IGLogWriter(
            s,
            log_filepath=out_log_path,
            store_vr=False,
            vr_robot=vr_agent,
            profiling_mode=profile,
            filter_objects=filter_objects,
        )
        log_writer.set_up_data_storage()

    for callback in start_callbacks:
        callback(vr_agent, log_reader)

    task_done = False
    satisfied_predicates = {}
    while log_reader.get_data_left_to_read():

        s.step(print_stats=profile)

        # Set camera each frame
        if mode == "vr":
            log_reader.set_replay_camera(s)

        for callback in step_callbacks:
            callback(vr_agent, log_reader)

        # Get relevant VR action data and update VR agent
        vr_agent.update(log_reader.get_agent_action("vr_robot"))

        if not disable_save:
            log_writer.process_frame()

    demo_statistics = {}
    for callback in end_callbacks:
        callback(vr_agent, log_reader)

    s.disconnect()

    demo_statistics = {
        "task_done": task_done,
        "satisfied_predicates": satisfied_predicates,
        "total_frame_num": log_reader.total_frame_num,
    }
    return demo_statistics


def main():
    args = parse_args()
    agent_metrics = AgentMetric()
    replay_demo(
        args.vr_log_path,
        out_log_path=args.vr_replay_log_path,
        step_callbacks=[agent_metrics.step_callback],
        disable_save=args.disable_save,
        frame_save_path=args.frame_save_path,
        mode=args.mode,
        profile=args.profile,
    )
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    main()
