"""
  Script to postprocess iGibson exported data to SUN RGB-D compatible data format.
"""

import argparse
import json
import os
from concurrent import futures as futures
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# Objects with (# cloud point) / (surface area) lower than this threshold will be filtered out.
# TODO: find a better filtering method.
CLOUD_POINT_RATIO_LOWER_THRESHOLD = 800


def generate_sunrgbd_data(data_dir, out_dir, skip_existing=True, max_workers=4):
    def process_single_demo(demo):
        print("Processing demo: {}".format(demo))

        try:
            # Read data.
            f = h5py.File(os.path.join(data_dir, demo + "_data.hdf5"), "r")
            colors = np.array(f["pointcloud"]["colors"])
            points = np.array(f["pointcloud"]["points"])
            bbox = np.array(f["bbox"])

            # Process each extracted frame of the demo replay.
            for frame_idx in range(len(points)):
                # Get image.
                cur_color = colors[frame_idx]
                img = Image.fromarray((cur_color * 255).astype(np.uint8), "RGB")

                # Get depth.
                cur_points = points[frame_idx]
                points_mask = cur_points[:, :, -1] == 1
                fileterd_points = cur_points[points_mask][:, :3]
                depth = np.concatenate((cur_points[points_mask][:, :3], cur_color[points_mask]), axis=-1)

                # Elements in bbox: body id, category id, 2d top left, 2d extent, 3d center, z-axis rotation, 3d extent.
                class_ids = bbox[frame_idx, :, 1:2]
                camera_frame_center = bbox[frame_idx, :, 6:9]
                extend_3d = bbox[frame_idx, :, 10:13]

                # Identify objects to keep.
                xyz_min = camera_frame_center - extend_3d / 2
                xyz_max = camera_frame_center + extend_3d / 2
                surface_area = (
                    extend_3d[:, 0] * extend_3d[:, 1]
                    + extend_3d[:, 1] * extend_3d[:, 2]
                    + extend_3d[:, 0] * extend_3d[:, 2]
                ) * 2
                keep_idx = []
                for class_idx, class_id in enumerate(class_ids.flatten()):
                    # Skip walls, floors and ceilings.
                    # ID 0-6: background, robots, user_added_objs, scene_objs, walls, floors, ceilings.
                    if class_id <= 6:
                        continue
                    point_cnt = (
                        np.all(
                            (fileterd_points >= xyz_min[class_idx]) & (fileterd_points <= xyz_max[class_idx]),
                            axis=1,
                        )
                    ).sum()
                    ratio = point_cnt / surface_area[class_idx]
                    # Skip objects with too few cloud points inside.
                    if ratio > CLOUD_POINT_RATIO_LOWER_THRESHOLD:
                        keep_idx.append(class_idx)

                if not keep_idx:
                    continue

                # Write label.
                class_id = bbox[frame_idx, keep_idx, 1:2]
                bb_top_left = bbox[frame_idx, keep_idx, 2:4]
                extend_2d = bbox[frame_idx, keep_idx, 4:6]
                camera_frame_center = bbox[frame_idx, keep_idx, 6:9]
                z_rotation = bbox[frame_idx, keep_idx, 9:10]
                extend_3d = bbox[frame_idx, keep_idx, 10:13]

                label = np.hstack(
                    (
                        class_id.astype(int),
                        bb_top_left[:, ::-1],  # top_left to left_top
                        extend_2d[:, ::-1],
                        camera_frame_center,
                        extend_3d[:, (1, 0, 2)] / 2,  # (w,l,h) = (dy/2,dx/2,dz/2)
                        np.cos(z_rotation),
                        np.sin(z_rotation),
                    )
                )

                # Save output files.
                img.save(os.path.join(out_dir, "image", "{}_{}.jpg".format(demo, frame_idx)))
                np.savetxt(
                    os.path.join(out_dir, "depth", "{}_{}.txt".format(demo, frame_idx)),
                    depth,
                    fmt="%1.4f",
                )
                np.savetxt(
                    os.path.join(out_dir, "label", "{}_{}.txt".format(demo, frame_idx)),
                    label,
                    fmt=(
                        "%d",
                        "%d",
                        "%d",
                        "%d",
                        "%d",
                        "%f",
                        "%f",
                        "%f",
                        "%f",
                        "%f",
                        "%f",
                        "%f",
                        "%f",
                    ),
                )

            with open(os.path.join(out_dir, "flag", "{}.txt".format(demo)), "w") as flag_f:
                flag_f.write(str(len(points)))

            # Close data file.
            f.close()
        except Exception as e:
            print("Demo {} failed with exception: {}".format(demo, e))

    # Create output folders.
    depth_dir = os.path.join(out_dir, "depth")
    image_dir = os.path.join(out_dir, "image")
    label_dir = os.path.join(out_dir, "label")
    flag_dir = os.path.join(out_dir, "flag")
    Path(depth_dir).mkdir(parents=True, exist_ok=True)
    Path(image_dir).mkdir(parents=True, exist_ok=True)
    Path(label_dir).mkdir(parents=True, exist_ok=True)
    Path(flag_dir).mkdir(parents=True, exist_ok=True)

    demo_list = []
    for filename in os.listdir(data_dir):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join(data_dir, filename), "r") as f:
            metadata = json.load(f)
            if metadata["failed"]:
                continue
        demo = os.path.splitext(filename)[0]
        flag_path = os.path.join(flag_dir, "{}.txt".format(demo))
        if skip_existing and os.path.exists(flag_path):
            continue
        demo_list.append(demo)

    print("Number of demos: {}".format(len(demo_list)))
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        infos = executor.map(process_single_demo, demo_list)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract SUN RGB-D data from exported iGibson data in data_dir.")
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Directory containing exported iGibson data.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to store the output data.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print(args)

    generate_sunrgbd_data(args.data_dir, args.out_dir, skip_existing=False, max_workers=8)


if __name__ == "__main__":
    main()
