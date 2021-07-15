import argparse
import glob
import os

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Script to generate manifest for batch replay demos")
    parser.add_argument(
        "--demo_directory", type=str, required=True, help="Plain text file consisting of list of demos to replay"
    )
    parser.add_argument(
        "--split", type=int, default=0, help="Number of times to split the manifest for distributing replay"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    demos = glob.glob(os.path.join(args.demo_directory, "*.hdf5"))
    filtered_demos = [demo for demo in demos if "replay" not in demo]
    series = {"demos": filtered_demos}
    pd.DataFrame(series).to_csv("manifest.csv")
    if args.split > 1:
        for idx, split in enumerate(np.array_split(filtered_demos, args.split)):
            pd.DataFrame({"demos": split}).to_csv("manifest_{}.csv".format(idx))


if __name__ == "__main__":
    main()
