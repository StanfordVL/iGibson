import argparse
import os

import igibson
from igibson.examples.behavior.behavior_demo_batch import behavior_demo_batch
from igibson.scene_graphs.graph_builder import SceneGraphBuilderWithVisualization


def parse_args():
    testdir = os.path.join(igibson.ig_dataset_path, "tests")
    manifest = os.path.join(testdir, "test_manifest.txt")
    parser = argparse.ArgumentParser(description="Build live scene graphs from BEHAVIOR demos in manifest.")
    parser.add_argument(
        "--demo_root", type=str, default=testdir, help="Directory containing demos listed in the manifest."
    )
    parser.add_argument(
        "--log_manifest", type=str, default=manifest, help="Plain text file consisting of list of demos to replay."
    )
    parser.add_argument("--out_dir", type=str, default=testdir, help="Directory to store results in.")
    return parser.parse_args()


def main():
    args = parse_args()

    def get_scene_graph_callbacks(demo_name, out_dir):
        path = os.path.join(out_dir, demo_name)
        if not os.path.exists(path):
            os.mkdir(path)
        graph_builder = SceneGraphBuilderWithVisualization(
            show_window=False, out_path=path, only_true=True, realistic_positioning=True
        )

        return (
            [graph_builder.start],
            [graph_builder.step],
            [],
            [],
        )

    behavior_demo_batch(
        args.demo_root,
        args.log_manifest,
        args.out_dir,
        get_scene_graph_callbacks,
        image_size=(640, 720),
        ignore_errors=False,
        debug_display=False,
    )


if __name__ == "__main__":
    main()
