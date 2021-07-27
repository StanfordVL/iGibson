import argparse
import glob
import os
import shlex
import subprocess

import pandas as pd
import tqdm

import igibson


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("log_manifest", type=str, help="Plain text file consisting of list of demos to replay.")
    # args = parser.parse_args()
    demos_path = r"C:\Users\cgokmen\Stanford Drive\BEHAVIOR resources\New NeurIPS Demos"
    segmentations_path = r"C:\Users\cgokmen\Stanford Drive\BEHAVIOR resources\segmentation_results"
    results_path = r"C:\Users\cgokmen\Stanford Drive\BEHAVIOR resources\mp_replay_results\examples"

    # Load the demo to get info
    # for demo_fullpath in tqdm.tqdm(list(glob.glob(os.path.join(segmentations_path, "*.json")))):
    for demo in ["putting_leftovers_away_0_Pomaria_1_int_2021-06-03_14-32-54"]:
        # demo = os.path.splitext(os.path.basename(demo_fullpath))[0]
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

        print(" ".join([shlex.quote(arg) for arg in args]))

        # with open(log_path, "w") as log_file:
        #     tqdm.tqdm.write("Processing %s" % demo)
        #     subprocess.run(args, stdout=log_file, stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
