import argparse
import os

import h5py
import numpy as np

import igibson
from igibson import object_states
from igibson.examples.behavior.behavior_demo_batch import behavior_demo_batch
from igibson.object_states.utils import get_center_extent
from igibson.utils.constants import MAX_INSTANCE_COUNT, SemanticClass


def parse_args():
    testdir = os.path.join(igibson.ig_dataset_path, "tests")
    manifest = os.path.join(testdir, "test_manifest.txt")
    parser = argparse.ArgumentParser(description="Extract ImVoteNet training data from BEHAVIOR demos in manifest.")
    parser.add_argument(
        "--demo_root", type=str, default=testdir, help="Directory containing demos listed in the manifest."
    )
    parser.add_argument(
        "--log_manifest", type=str, default=manifest, help="Plain text file consisting of list of demos to replay."
    )
    parser.add_argument("--out_dir", type=str, default=testdir, help="Directory to store results in.")
    return parser.parse_args()


class PointCloudExtractor(object):
    def __init__(self, h5py_file):
        self.h5py_file = h5py_file
        self.points = None
        self.colors = None
        self.categories = None
        self.instances = None

    def start_callback(self, igbhvr_act_inst, log_reader):
        # Create the dataset
        renderer = igbhvr_act_inst.simulator.renderer
        w = renderer.width
        h = renderer.height
        n_frames = log_reader.total_frame_num
        self.points = self.h5py_file.create_dataset(
            "/pointcloud/points", (n_frames, h, w, 4), dtype=np.float32, compression="gzip", compression_opts=9
        )
        self.colors = self.h5py_file.create_dataset(
            "/pointcloud/colors", (n_frames, h, w, 3), dtype=np.float32, compression="gzip", compression_opts=9
        )
        self.categories = self.h5py_file.create_dataset(
            "/pointcloud/categories", (n_frames, h, w), dtype=np.int32, compression="gzip", compression_opts=9
        )
        self.instances = self.h5py_file.create_dataset(
            "/pointcloud/instances", (n_frames, h, w), dtype=np.int32, compression="gzip", compression_opts=9
        )

    def step_callback(self, igbhvr_act_inst, _):
        # TODO: Check how this compares to the outputs of SUNRGBD. Currently we're just taking the robot FOV.
        renderer = igbhvr_act_inst.simulator.renderer
        rgb, seg, ins_seg, threed = renderer.render_robot_cameras(modes=("rgb", "seg", "ins_seg", "3d"))

        # Get rid of extra dimensions on segmentations
        seg = seg[:, :, 0].astype(int)
        ins_seg = np.round(ins_seg[:, :, 0] * MAX_INSTANCE_COUNT).astype(int)
        id_seg = renderer.get_pb_ids_for_instance_ids(ins_seg)

        self.points[igbhvr_act_inst.simulator.frame_count] = threed.astype(np.float32)
        self.colors[igbhvr_act_inst.simulator.frame_count] = rgb[:, :, :3].astype(np.float32)
        self.categories[igbhvr_act_inst.simulator.frame_count] = seg.astype(np.int32)
        self.instances[igbhvr_act_inst.simulator.frame_count] = id_seg.astype(np.int32)


class BBox2DExtractor(object):
    def __init__(self, h5py_file):
        self.h5py_file = h5py_file
        self.bboxes = None
        self.cameraV = None
        self.cameraP = None

    def start_callback(self, _, log_reader):
        # Create the dataset
        n_frames = log_reader.total_frame_num
        # body id, category id, 2d top left, 2d extent
        self.bboxes = self.h5py_file.create_dataset("/bbox2d", (n_frames, MAX_INSTANCE_COUNT, 6), dtype=np.float32)
        self.cameraV = self.h5py_file.create_dataset("/cameraV", (n_frames, 4, 4), dtype=np.float32)
        self.cameraP = self.h5py_file.create_dataset("/cameraP", (n_frames, 4, 4), dtype=np.float32)

    def step_callback(self, igbhvr_act_inst, _):
        renderer = igbhvr_act_inst.simulator.renderer
        ins_seg = renderer.render_robot_cameras(modes="ins_seg")[0][:, :, 0]
        ins_seg = np.round(ins_seg * MAX_INSTANCE_COUNT).astype(int)
        id_seg = renderer.get_pb_ids_for_instance_ids(ins_seg)

        out = np.full((MAX_INSTANCE_COUNT, 6), -1, dtype=np.float32)
        filled_obj_idx = 0

        for body_id in np.unique(id_seg):
            if body_id == -1 or body_id not in igbhvr_act_inst.simulator.scene.objects_by_id:
                continue

            this_object_pixels_positions = np.argwhere(id_seg == body_id)

            bb_top_left = np.min(this_object_pixels_positions, axis=0)
            bb_bottom_right = np.max(this_object_pixels_positions, axis=0)

            # Get the object semantic class ID
            obj = igbhvr_act_inst.simulator.scene.objects_by_id[body_id]
            class_id = igbhvr_act_inst.simulator.class_name_to_class_id.get(obj.category, SemanticClass.SCENE_OBJS)

            # Record the results.
            out[filled_obj_idx, 0] = body_id
            out[filled_obj_idx, 1] = class_id
            out[filled_obj_idx, 2:4] = bb_top_left
            out[filled_obj_idx, 4:6] = bb_bottom_right - bb_top_left
            filled_obj_idx += 1

        # Add this frame's results to the overall results.
        self.bboxes[igbhvr_act_inst.simulator.frame_count] = out
        self.cameraV[igbhvr_act_inst.simulator.frame_count] = renderer.V
        self.cameraP[igbhvr_act_inst.simulator.frame_count] = renderer.P

        # Uncomment this to debug the 2d bounding box setup (set a breakpoint on plt.show())
        # img = igbhvr_act_inst.simulator.renderer.render_robot_cameras(modes=("rgb"))[0]
        # plt.imshow(img[:, :, :3])
        # for bb in bbs:
        #     # Note that the Rectangle expects x, y coordinates but we have y, x
        #     plt.gca().add_patch(Rectangle(tuple(np.flip(bb[1])), bb[2][1], bb[2][0],
        #                                   edgecolor='red',
        #                                   facecolor='none',
        #                                   lw=4))
        # plt.show()


class BBox3DExtractor(object):
    def __init__(self, h5py_file):
        self.h5py_file = h5py_file
        self.bboxes = None

    def start_callback(self, _, log_reader):
        # Create the dataset
        n_frames = log_reader.total_frame_num
        # body id, category id, 3d center, quat(4) orientation, 3d extent
        self.bboxes = self.h5py_file.create_dataset("/bbox3d", (n_frames, MAX_INSTANCE_COUNT, 12), dtype=np.float32)

    def step_callback(self, igbhvr_act_inst, _):
        out = np.full((MAX_INSTANCE_COUNT, 12), -1, dtype=np.float32)
        filled_obj_idx = 0

        for obj in igbhvr_act_inst.simulator.scene.get_objects():
            # Check if the object is in sight.
            # TODO: What do we need to do if the bbox is partially in the robot FOV? truncate?
            if not obj.states[object_states.InFOVOfRobot].get_value():
                continue

            # Get the center and extent
            center, extent = get_center_extent(obj.states)

            # Assume that the extent is when the object is axis-aligned.
            # Convert that pose to the camera frame too.
            # TODO: This is in camera frame, not upright camera frame. Easy fix - but should we do it here?
            pose = np.concatenate([center, np.array([0, 0, 0, 1])])
            transformed_pose = igbhvr_act_inst.simulator.renderer.transform_pose(pose)
            center_cam = transformed_pose[:3]
            orientation_cam = transformed_pose[3:]

            # Get the object semantic class ID
            class_id = igbhvr_act_inst.simulator.class_name_to_class_id.get(obj.category, SemanticClass.SCENE_OBJS)

            # Record the results.
            out[filled_obj_idx, 0] = obj.get_body_id()
            out[filled_obj_idx, 1] = class_id
            out[filled_obj_idx, 2:5] = center_cam
            out[filled_obj_idx, 5:9] = orientation_cam
            out[filled_obj_idx, 9:12] = extent
            filled_obj_idx += 1

        self.bboxes[igbhvr_act_inst.simulator.frame_count] = out


def main():
    args = parse_args()

    def get_imvotenet_callbacks(demo_name, out_dir):
        path = os.path.join(out_dir, demo_name + "_data.h5py")
        h5py_file = h5py.File(path, "w")
        extractors = [PointCloudExtractor(h5py_file), BBox3DExtractor(h5py_file), BBox2DExtractor(h5py_file)]

        return (
            [extractor.start_callback for extractor in extractors],
            [extractor.step_callback for extractor in extractors],
            [lambda a, b: h5py_file.close()],  # Close the file once we're done.
            [],
        )

    # TODO: Set resolution to match model.
    behavior_demo_batch(
        args.demo_root,
        args.log_manifest,
        args.out_dir,
        get_imvotenet_callbacks,
        image_size=(480, 480),
        ignore_errors=False,
    )


if __name__ == "__main__":
    main()
