import argparse

from igibson.examples.behavior.behavior_demo_batch import behavior_demo_batch


def parse_args():
    parser = argparse.ArgumentParser(description="Collect metrics from BEHAVIOR demos in manifest.")
    parser.add_argument("--demo_root", type=str, help="Directory containing demos listed in the manifest.")
    parser.add_argument("--log_manifest", type=str, help="Plain text file consisting of list of demos to replay.")
    parser.add_argument("--out_dir", type=str, help="Directory to store results in.")
    return parser.parse_args()


def main():
    args = parse_args()

    def get_metrics_callbacks():
        return ( [], [], [], [],)

    behavior_demo_batch(args.demo_root, args.log_manifest, args.out_dir, get_metrics_callbacks)


if __name__ == "__main__":
    main()
