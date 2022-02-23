import argparse
import glob
import logging
import os

import numpy as np
import pandas as pd

import igibson


def parse_args(defaults=False):
    args_dict = dict()
    args_dict["demo_dir"] = os.path.join(igibson.ig_dataset_path, "tests")
    args_dict["split"] = 0
    if not defaults:
        parser = argparse.ArgumentParser(description="Script to generate manifest for batch replay demos")
        parser.add_argument(
            "--demo_dir",
            type=str,
            required=True,
            help="Directory containing a group of demos to replay with a manifest",
        )
        parser.add_argument(
            "--split",
            type=int,
            default=args_dict["split"],
            help="Number of times to split the manifest for distributing replay",
        )
        args = parser.parse_args()
        args_dict["demo_dir"] = args.demo_dir
        args_dict["split"] = args.split
    return args_dict


def main(selection="user", headless=False, short_exec=False):
    """
    Generates a manifest file for batch processing of BEHAVIOR demos
    """
    logging.getLogger().setLevel(logging.INFO)
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    defaults = selection == "random" and headless and short_exec
    args_dict = parse_args(defaults=defaults)

    demos = glob.glob(os.path.join(args_dict["demo_dir"], "*.hdf5"))
    logging.info("Demos to add to the manifest: {}".format(demos))
    filtered_demos = [demo for demo in demos if "replay" not in demo]
    logging.info("Demos to add to the manifest after removing replays: {}".format(demos))
    series = {"demos": filtered_demos}
    output_folder = os.path.join(igibson.ig_dataset_path, "tests")
    pd.DataFrame(series).to_csv(os.path.join(output_folder, "manifest.csv"))
    if args_dict["split"] > 1:
        for idx, split in enumerate(np.array_split(filtered_demos, args_dict["split"])):
            pd.DataFrame({"demos": split}).to_csv(os.path.join(output_folder, "manifest_{}.csv".format(idx)))

    logging.info("Manifest(s) created")


RUN_AS_TEST = False  # Change to True to run this example in test mode
if __name__ == "__main__":
    if RUN_AS_TEST:
        main(selection="random", headless=True, short_exec=True)
    else:
        main()
