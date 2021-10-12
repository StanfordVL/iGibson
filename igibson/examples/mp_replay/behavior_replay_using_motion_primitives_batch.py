import argparse
import glob
import os
import subprocess

import tqdm

import igibson


def parse_args():
    parser = argparse.ArgumentParser(description="Script to batch-replay segmented demos using motion primitives.")
    parser.add_argument("demo_directory", type=str, help="Path to directory containing demos")
    parser.add_argument("segmentation_directory", type=str, help="Path to directory containing demo segmentations")
    parser.add_argument("results_directory", type=str, help="Path to directory to store results in")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the demo to get info
    for demo_fullpath in tqdm.tqdm(list(glob.glob(os.path.join(args.segmentation_directory, "*.json")))):
        demo = os.path.splitext(os.path.basename(demo_fullpath))[0]
        if "replay" in demo:
            continue

        demo_path = os.path.join(args.demo_directory, demo + ".hdf5")
        segmentation_path = os.path.join(args.segmentation_directory, demo + ".json")
        output_path = os.path.join(args.results_directory, demo + ".json")
        log_path = os.path.join(args.results_directory, demo + ".log")

        if os.path.exists(output_path):
            continue

        # Batch me
        script_path = os.path.join(igibson.example_path, "behavior", "behavior_replay_using_motion_primitives.py")
        command = ["python", script_path, demo_path, segmentation_path, output_path]

        with open(log_path, "w") as log_file:
            tqdm.tqdm.write("Processing %s" % demo)
            subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
