import argparse
import glob
import os
import shlex
import subprocess

import pandas as pd
import tqdm

import igibson


def main():
    # TODO(replayMP): Update this path
    path_of_gdrive_behavior_resources_dir = r"/home/michael/Repositories/lab/iGibson"
    demos_path = os.path.join(path_of_gdrive_behavior_resources_dir, r"demos")
    segmentations_path = os.path.join(path_of_gdrive_behavior_resources_dir, r"segmentation_results")
    results_path = os.path.join(path_of_gdrive_behavior_resources_dir, r"mp_replay_results_2/examples")

    # Load the demo to get info
    for demo_fullpath in tqdm.tqdm(list(glob.glob(os.path.join(segmentations_path, "*.json")))):
        demo = os.path.splitext(os.path.basename(demo_fullpath))[0]
        if "replay" in demo:
            continue

        demo_path = os.path.join(demos_path, demo + ".hdf5")
        segmentation_path = os.path.join(segmentations_path, demo + ".json")
        output_path = os.path.join(results_path, demo + ".json")
        log_path = os.path.join(results_path, demo + ".log")

        if os.path.exists(output_path):
            continue

        # Batch me
        script_path = os.path.join(igibson.example_path, "behavior", "behavior_replay_using_motion_primitives.py")
        args = ["python", script_path, demo_path, segmentation_path, output_path]

        # print(" ".join([shlex.quote(arg) for arg in args]))

        with open(log_path, "w") as log_file:
            tqdm.tqdm.write("Processing %s" % demo)
            subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
