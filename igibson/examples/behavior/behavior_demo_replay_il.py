"""
Main BEHAVIOR demo replay entrypoint
"""

import argparse
import datetime
import json
import os
import pprint
import random

# from torch.utils.tensorboard import SummaryWriter
import shutil
import time

import bddl
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset

import igibson
from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.simulator import Simulator
from igibson.utils.git_utils import project_git_info
from igibson.utils.ig_logging import IGLogReader, IGLogWriter
from igibson.utils.utils import parse_str_config


class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, arm_action_size):
        super(Model, self).__init__()
        assert num_layers > 0
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.arm_action_head = nn.Linear(hidden_size, arm_action_size)

    def forward(self, x):
        x = self.layers(x)
        arm_action = self.arm_action_head(x)
        return arm_action


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

    parser.add_argument("--rewind", type=int)

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
    rewind=None,
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
        image_width=1280,
        image_height=720,
    )

    igbhvr_act_inst = iGBEHAVIORActivityInstance(task, task_id)
    igbhvr_act_inst.initialize_simulator(
        simulator=s,
        scene_id=scene,
        scene_kwargs={
            "urdf_file": "{}_task_{}_{}_0_fixed_furniture".format(scene, task, task_id),
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
            out_log_path = "{}_{}_{}_{}_replay.hdf5".format(task, task_id, scene, timestamp)

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

    task_done = False

    # initialize model
    model = Model(
        input_size=44,
        hidden_size=64,
        num_layers=5,
        arm_action_size=12,
    )
    model = torch.nn.DataParallel(model).cuda()

    resume = "/home/fei/Development/iGibson/igibson/examples/behavior/bc_results/ckpt/checkpoint_10.pth.tar"

    if os.path.isfile(resume):
        print("=> loading checkpoint '{}'".format(resume))
        # if args.gpu is None:
        checkpoint = torch.load(resume)
        # else:
        #    # Map model to be loaded to specified single gpu.
        #    loc = 'cuda:{}'.format(args.gpu)
        #    checkpoint = torch.load(args.resume, map_location=loc)
        # args.start_epoch = checkpoint['epoch']
        # best_l1 = checkpoint['best_l1']
        # if args.gpu is not None:
        #    # best_acc1 may be from a checkpoint from a different GPU
        #    best_l1 = best_l1.to(args.gpu)
        model.load_state_dict(checkpoint["state_dict"])
        # optimizer.load_state_dict(checkpoint['optimizer'])
        input_mean = checkpoint["input_mean"]
        input_std = checkpoint["input_std"]

        print("=> loaded checkpoint '{}' (epoch {})".format(resume, checkpoint["epoch"]))
    else:
        print("=> no checkpoint found at '{}'".format(resume))

    while log_reader.get_data_left_to_read():

        igbhvr_act_inst.simulator.step(print_stats=profile)
        task_done, _ = igbhvr_act_inst.check_success()

        # Set camera each frame
        if mode == "vr":
            log_reader.set_replay_camera(s)

        # save_fn = "{}/frame_{:04d}.jpg".format(frame_save_path, log_reader.frame_counter)
        # rgb_frame = igbhvr_act_inst.simulator.renderer.render_robot_cameras()[0][:,:,:3]
        # if log_reader.frame_counter > log_reader.total_frame_num - rewind:
        #     rgb_frame[:20, :, 0] = 0
        #     rgb_frame[:20, :, 1] = 1
        #     rgb_frame[:20, :, 2] = 0
        # else:
        #     rgb_frame[:20, :, 0] = 0
        #     rgb_frame[:20, :, 1] = 0
        #     rgb_frame[:20, :, 2] = 1
        #
        # image = Image.fromarray((rgb_frame * 255).astype(np.uint8))
        # image.save(save_fn)

        left_hand_local_pos = vr_agent.parts["left_hand"].local_pos
        left_hand_local_orn = vr_agent.parts["left_hand"].local_orn
        right_hand_local_pos = vr_agent.parts["right_hand"].local_pos
        right_hand_local_orn = vr_agent.parts["right_hand"].local_orn
        left_hand_trigger_fraction = [vr_agent.parts["left_hand"].trigger_fraction]
        right_hand_trigger_fraction = [vr_agent.parts["right_hand"].trigger_fraction]

        proprioception = np.concatenate(
            (
                left_hand_local_pos,
                left_hand_local_orn,
                right_hand_local_pos,
                right_hand_local_orn,
                left_hand_trigger_fraction,
                right_hand_trigger_fraction,
            ),
        )
        keys = ["1", "62", "8", "98"]
        tracked_objects = ["floor.n.01_1", "caldron.n.01_1", "table.n.02_1", "agent.n.01_1"]
        task_obs = []

        for obj in tracked_objects:
            pos, orn = igbhvr_act_inst.object_scope[obj].get_position_orientation()
            task_obs.append(np.array(pos))
            task_obs.append(np.array(orn))

        task_obs = np.concatenate(task_obs)
        # task_obs[21 + 2] += 0.6
        # from IPython import embed;
        # embed()

        agent_input = np.concatenate((proprioception, task_obs))
        agent_input = (agent_input - input_mean) / (input_std + 1e-10)
        agent_input = agent_input[None, :].astype(np.float32)

        with torch.no_grad():
            pred_action = model(torch.from_numpy(agent_input)).cpu().numpy()[0]

        # print('pred action', pred_action)
        # print('true action', log_reader.get_agent_action('vr_robot'))
        true_action = log_reader.get_agent_action("vr_robot")

        if log_reader.frame_counter > log_reader.total_frame_num - rewind:
            true_action[12:18] = pred_action[:6]
            true_action[20:26] = pred_action[6:]
            print("using pred action")

        for callback in step_callbacks:
            callback(igbhvr_act_inst, log_reader)

        # Get relevant VR action data and update VR agent
        # vr_agent.apply_action(log_reader.get_agent_action("vr_robot"))
        vr_agent.apply_action(true_action)

        if not disable_save:
            log_writer.process_frame()

    print("Demo was succesfully completed: ", task_done)

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
        rewind=args.rewind,
    )


if __name__ == "__main__":
    main()
