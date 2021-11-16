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
from igibson.examples.behavior.behavior_demo_replay import replay_demo
from igibson.metrics.dataset import DatasetMetric
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_vr import VrSettings
from igibson.sensors.vision_sensor import VisionSensor
from igibson.simulator import Simulator
from igibson.utils.git_utils import project_git_info
from igibson.utils.ig_logging import IGLogReader
from igibson.utils.utils import parse_str_config


def save_episode(in_log_path, dataset_metric):
    activity = IGLogReader.read_metadata_attr(in_log_path, "/metadata/atus_activity")
    activity_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/activity_definition")
    scene = IGLogReader.read_metadata_attr(in_log_path, "/metadata/scene_id")
    physics_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/physics_timestep")
    render_timestep = IGLogReader.read_metadata_attr(in_log_path, "/metadata/render_timestep")
    instance_id = IGLogReader.read_metadata_attr(in_log_path, "/metadata/instance_id")
    urdf_file = IGLogReader.read_metadata_attr(in_log_path, "/metadata/urdf_file")

    episode_identifier = "_".join(os.path.splitext(in_log_path)[0].split("_")[-2:])

    episode_out_log_path = "processed_hdf5s/{}_{}_{}_{}_episode.hdf5".format(
        activity, activity_id, scene, episode_identifier
    )
    hf = h5py.File(episode_out_log_path, "w")
    hf.attrs["/metadata/physics_timestep"] = physics_timestep
    hf.attrs["/metadata/render_timestep"] = render_timestep
    hf.attrs["/metadata/activity"] = activity
    hf.attrs["/metadata/activity_id"] = activity_id
    hf.attrs["/metadata/scene_id"] = scene

    for key, value in dataset_metric.gather_results().items():
        if key == "action":
            dtype = np.float64
        else:
            dtype = np.float32
        hf.create_dataset(key, data=np.stack(value), dtype=dtype, compression="lzf")

    hf.close()


def generate_imitation_dataset(demo_root, log_manifest, out_dir, config_file, skip_existing=True, save_frames=False):
    """
    Execute imitation dataset generation on a batch of BEHAVIOR demos.

    @param demo_root: Directory containing the demo files listed in the manifests.
    @param log_manifest: The manifest file containing list of BEHAVIOR demos to batch over.
    @param out_dir: Directory to store results in.
    @param config_file: environment config file
    @param skip_existing: Whether demos with existing output logs should be skipped.
    @param save_frames: Whether the demo's frames should be saved alongside statistics.
    """
    logger = logging.getLogger()
    logger.disabled = True

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
            dataset = DatasetMetric()
            demo_information = replay_demo(
                in_log_path=demo_path,
                frame_save_path=curr_frame_save_path,
                mode="headless",
                config_file=config_file,
                verbose=False,
                image_size=(128, 128),
                start_callbacks=[dataset.start_callback],
                step_callbacks=[dataset.step_callback],
                end_callbacks=[dataset.end_callback],
            )
            demo_information["failed"] = False
            demo_information["filename"] = Path(demo).name
            save_episode(demo_path, dataset)

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
    parser.add_argument("--vr_log_path", type=str, help="Path (and filename) of vr log to replay")
    parser.add_argument("--config", help="which config file to use [default: use yaml files in examples/configs]")

    return parser.parse_args()


def main():
    args = parse_args()
    generate_imitation_dataset(args.demo_root, args.log_manifest, args.out_dir, args.config)


if __name__ == "__main__":
    main()
