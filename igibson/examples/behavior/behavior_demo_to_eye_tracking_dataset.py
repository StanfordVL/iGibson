import argparse
import json
import os

import h5py
import numpy as np
import pybullet as p

import igibson
from igibson.examples.behavior.behavior_demo_batch import behavior_demo_batch
from igibson.objects.articulated_object import URDFObject
from igibson.utils import utils
from igibson.utils.constants import MAX_INSTANCE_COUNT, SemanticClass


def parse_args():
    testdir = os.path.join(igibson.ig_dataset_path, "tests")
    manifest = os.path.join(testdir, "test_manifest.txt")
    parser = argparse.ArgumentParser(description="Extract eye tracking data from BEHAVIOR demos in manifest.")
    parser.add_argument(
        "--demo_root", type=str, default=testdir, help="Directory containing demos listed in the manifest."
    )
    parser.add_argument(
        "--log_manifest", type=str, default=manifest, help="Plain text file consisting of list of demos to replay."
    )
    parser.add_argument("--out_dir", type=str, default=testdir, help="Directory to store results in.")
    return parser.parse_args()


class EyeTrackingExtractor(object):
    def __init__(self, h5py_file):
        self.h5py_file = h5py_file
        self.directly_attended = None
        self.approximately_attended = None

    def start_callback(self, igbhvr_act_inst, log_reader):
        # Create the dataset
        num_bodies = p.getNumBodies()
        num_bytes = -(num_bodies // -8)  # This is rounded up.
        n_frames = log_reader.total_frame_num
        self.directly_attended = np.full(n_frames, -1, dtype=np.int16)
        self.approximately_attended = np.zeros((n_frames, num_bytes), dtype=np.uint8)

        scene = igbhvr_act_inst.simulator.scene
        self.h5py_file.attrs["body_id_to_category"] = json.dump(
            {bid: scene.objects_by_id[bid].category for bid in range(num_bodies) if bid in scene.objects_by_id}
        )
        self.h5py_file.attrs["body_id_to_name"] = json.dump(
            {bid: scene.objects_by_id[bid].name for bid in range(num_bodies) if bid in scene.objects_by_id}
        )

    def step_callback(self, igbhvr_act_inst, log_reader):
        frame_count = igbhvr_act_inst.simulator.frame_count
        robot = igbhvr_act_inst.simulator.robots[0]
        eye_data = log_reader.get_vr_data().query("eye_data")
        if eye_data[0] != -1:
            eye_pos = eye_data[1:4]
            eye_dir = eye_data[4:7]
            eye_dir /= np.linalg.norm(eye_dir)

            # Get an up axis.
            up = np.array([0, 0, 1])
            eye_up = up - (np.dot(eye_dir, up) * eye_dir)
            eye_up /= np.linalg.norm(eye_up)

            eye_y = np.cross(eye_up, eye_dir)
            eye_mat = np.array([eye_dir, eye_y, eye_up]).T

            # Fit a rotation.
            eye_orn = Rotation.from_matrix(eye_mat).as_quat()
            robot.parts["eye"].set_position_orientation(eye_pos, eye_orn)

            # Actually render & grab the directly-attended object.
            img_w, img_h = igbhvr_act_inst.simulator.image_width, igbhvr_act_inst.simulator.image_height
            gaze2D = (img_w // 2, img_h // 2)
            seg = robot.render_camera_image(modes="ins_seg")[0][:, :, 0]
            seg = np.transpose(np.round(seg * MAX_INSTANCE_COUNT).astype(int))
            self.directly_attended[frame_count] = seg[gaze2D]

            # Get the approximately-attended objects.
            radius = 4 * img_w // 120  # 120: FOV
            row_min, row_max = (
                gaze2D[0] - radius,
                gaze2D[0] + radius,
            )
            col_min, col_max = gaze2D[1] - radius, gaze2D[1] + radius
            sub_seg = seg[row_min:row_max, col_min:col_max]
            attended_ids = igbhvr_act_inst.simulator.renderer.get_pb_ids_for_instance_ids(sub_seg)
            attended_ids = set(np.unique(attended_ids)) - {-1}

            # Convert the approximately attended objects to a bitvector.
            attended_id_map = np.zeros(p.getNumBodies(), dtype=np.bool)
            attended_id_map[attended_ids] = True
            attended_id_bitvec = np.packbits(attended_id_map)
            self.approximately_attended[frame_count] = attended_id_bitvec

    def end_callback(self, *args):
        directly_attended = self.h5py_file.create_dataset(
            "/directly_attended",
            self.directly_attended.shape,
            dtype=np.uint16,
            compression="lzf",
        )
        directly_attended[:] = self.directly_attended

        approximately_attended = self.h5py_file.create_dataset(
            "/approximately_attended",
            self.approximately_attended.shape,
            dtype=np.uint8,
            compression="lzf",
        )
        approximately_attended[:] = self.approximately_attended


def main():
    args = parse_args()

    print(args)

    def get_eye_tracking_callbacks(demo_name, out_dir):
        path = os.path.join(out_dir, demo_name + "_data.h5py")
        h5py_file = h5py.File(path, "w")
        extractors = [EyeTrackingExtractor(h5py_file)]

        return (
            [extractor.start_callback for extractor in extractors],
            [extractor.step_callback for extractor in extractors],
            [extractor.end_callback for extractor in extractors] + [lambda a, b: h5py_file.close()],
            [],
        )

    behavior_demo_batch(
        args.demo_root,
        args.log_manifest,
        args.out_dir,
        get_eye_tracking_callbacks,
        image_size=(480, 480),
        ignore_errors=True,
        debug_display=False,
    )


if __name__ == "__main__":
    main()
