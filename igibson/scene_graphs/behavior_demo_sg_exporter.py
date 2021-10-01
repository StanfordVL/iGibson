import argparse
import os

import h5py

import igibson
from igibson.examples.behavior.behavior_demo_batch import behavior_demo_batch
from igibson.scene_graphs.graph_exporter import SceneGraphExporter


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
    parser.add_argument("--num_frames_per_demo", type=int, default=None, help="Number of frames to save per demo.")
    return parser.parse_args()


def main():
    args = parse_args()

    def get_scene_graph_callbacks(demo_name, out_dir):
        path = os.path.join(out_dir, demo_name + "_sg_data.hdf5")
        h5py_file = h5py.File(path, "w")
        extractors = [SceneGraphExporter(h5py_file, full_obs=True, num_frames_to_save=args.num_frames_per_demo)]

        return (
            [extractor.start for extractor in extractors],
            [extractor.step for extractor in extractors],
            [lambda a, b: h5py_file.close()],
            [],
        )

    behavior_demo_batch(
        args.demo_root,
        args.log_manifest,
        args.out_dir,
        get_scene_graph_callbacks,
        image_size=(640, 720),
        ignore_errors=True,
        debug_display=False,
    )


if __name__ == "__main__":
    main()
