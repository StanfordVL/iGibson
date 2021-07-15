import argparse

from igibson.examples.behavior.behavior_demo_batch import behavior_demo_batch
from igibson.examples.behavior.behavior_demo_segmentation import get_default_segmentation_processors


def parse_args():
    parser = argparse.ArgumentParser(description="Collect metrics from BEHAVIOR demos in manifest.")
    parser.add_argument("demo_root", type=str, help="Directory containing demos listed in the manifest.")
    parser.add_argument("log_manifest", type=str, help="Plain text file consisting of list of demos to replay.")
    parser.add_argument("out_dir", type=str, help="Directory to store results in.")
    return parser.parse_args()


def main():
    args = parse_args()

    def get_segmentation_callbacks():
        # Create default segmentation processors.
        segmentation_processors = get_default_segmentation_processors()

        # Create a data callback that unifies the results from the
        # segmentation processors.
        def data_callback():
            return {"segmentations": {name: sp.serialize_segments() for name, sp in segmentation_processors.items()}}

        # Return all of the callbacks for a particular demo.
        return (
            [sp.start_callback for sp in segmentation_processors.values()],
            [sp.step_callback for sp in segmentation_processors.values()],
            [],
            [data_callback],
        )

    behavior_demo_batch(args.demo_root, args.log_manifest, args.out_dir, get_segmentation_callbacks)


if __name__ == "__main__":
    main()
