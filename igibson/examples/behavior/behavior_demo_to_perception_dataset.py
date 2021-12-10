import argparse
import itertools
import os

import h5py
import numpy as np
import pybullet as p
import trimesh

import igibson
from igibson.examples.behavior.behavior_demo_batch import behavior_demo_batch
from igibson.objects.articulated_object import URDFObject
from igibson.utils import utils
from igibson.utils.constants import MAX_INSTANCE_COUNT, SemanticClass

FRAME_BATCH_SIZE = 100
START_FRAME = 500
SUBSAMPLE_EVERY_N_FRAMES = 60
SAVE_CAMERA = False
DEBUG_DRAW = False


def is_subsampled_frame(frame_count):
    return frame_count >= START_FRAME and (frame_count - START_FRAME) % SUBSAMPLE_EVERY_N_FRAMES == 0


def frame_to_entry_idx(frame_count):
    return (frame_count - START_FRAME) // SUBSAMPLE_EVERY_N_FRAMES


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

    def start_callback(self, env, log_reader):
        # Create the dataset
        renderer = env.simulator.renderer
        w = renderer.width
        h = renderer.height
        n_frames = frame_to_entry_idx(log_reader.total_frame_num)
        self.points = self.h5py_file.create_dataset(
            "/pointcloud/points",
            (n_frames, h, w, 4),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(n_frames, FRAME_BATCH_SIZE), h, w, 4),
        )
        self.colors = self.h5py_file.create_dataset(
            "/pointcloud/colors",
            (n_frames, h, w, 3),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(n_frames, FRAME_BATCH_SIZE), h, w, 3),
        )
        self.categories = self.h5py_file.create_dataset(
            "/pointcloud/categories",
            (n_frames, h, w),
            dtype=np.int32,
            compression="lzf",
            chunks=(min(n_frames, FRAME_BATCH_SIZE), h, w),
        )
        self.instances = self.h5py_file.create_dataset(
            "/pointcloud/instances",
            (n_frames, h, w),
            dtype=np.int32,
            compression="lzf",
            chunks=(min(n_frames, FRAME_BATCH_SIZE), h, w),
        )

        self.create_caches(renderer)

    def create_caches(self, renderer):
        w = renderer.width
        h = renderer.height

        self.points_cache = np.zeros((FRAME_BATCH_SIZE, h, w, 4), dtype=np.float32)
        self.colors_cache = np.zeros((FRAME_BATCH_SIZE, h, w, 3), dtype=np.float32)
        self.categories_cache = np.zeros((FRAME_BATCH_SIZE, h, w), dtype=np.int32)
        self.instances_cache = np.zeros((FRAME_BATCH_SIZE, h, w), dtype=np.int32)

    def write_to_file(self, env):
        frame_count = frame_to_entry_idx(env.simulator.frame_count)
        new_lines = frame_count % FRAME_BATCH_SIZE
        if new_lines == 0:
            return

        start_pos = frame_count - new_lines
        self.points[start_pos:frame_count] = self.points_cache[:new_lines]
        self.colors[start_pos:frame_count] = self.colors_cache[:new_lines]
        self.categories[start_pos:frame_count] = self.categories_cache[:new_lines]
        self.instances[start_pos:frame_count] = self.instances_cache[:new_lines]

        self.create_caches(env.simulator.renderer)

    def step_callback(self, env, _):
        if not is_subsampled_frame(env.simulator.frame_count):
            return

        # TODO: Check how this compares to the outputs of SUNRGBD. Currently we're just taking the robot FOV.
        renderer = env.simulator.renderer
        rgb, seg, ins_seg, threed = renderer.render_robot_cameras(modes=("rgb", "seg", "ins_seg", "3d"))

        # Get rid of extra dimensions on segmentations
        seg = seg[:, :, 0].astype(int)
        ins_seg = np.round(ins_seg[:, :, 0] * MAX_INSTANCE_COUNT).astype(int)
        id_seg = renderer.get_pb_ids_for_instance_ids(ins_seg)

        frame_idx = frame_to_entry_idx(env.simulator.frame_count) % FRAME_BATCH_SIZE
        self.points_cache[frame_idx] = threed.astype(np.float32)
        self.colors_cache[frame_idx] = rgb[:, :, :3].astype(np.float32)
        self.categories_cache[frame_idx] = seg.astype(np.int32)
        self.instances_cache[frame_idx] = id_seg.astype(np.int32)

        if frame_idx == FRAME_BATCH_SIZE - 1:
            self.write_to_file(env)

    def end_callback(self, env, _):
        self.write_to_file(env)


class BBoxExtractor(object):
    def __init__(self, h5py_file):
        self.h5py_file = h5py_file
        self.bboxes = None

        if SAVE_CAMERA:
            self.cameraV = None
            self.cameraP = None

    def start_callback(self, _, log_reader):
        # Create the dataset
        n_frames = frame_to_entry_idx(log_reader.total_frame_num)
        # body id, category id, 2d top left, 2d extent, 3d center, 4d orientation quat, 3d extent
        self.bboxes = self.h5py_file.create_dataset(
            "/bbox2d",
            (n_frames, p.getNumBodies(), 16),
            dtype=np.float32,
            compression="lzf",
            chunks=(min(n_frames, FRAME_BATCH_SIZE), p.getNumBodies(), 16),
        )

        if SAVE_CAMERA:
            self.cameraV = self.h5py_file.create_dataset(
                "/cameraV",
                (n_frames, 4, 4),
                dtype=np.float32,
                compression="lzf",
                chunks=(min(n_frames, FRAME_BATCH_SIZE), 4, 4),
            )
            self.cameraP = self.h5py_file.create_dataset(
                "/cameraP",
                (n_frames, 4, 4),
                dtype=np.float32,
                compression="lzf",
                chunks=(min(n_frames, FRAME_BATCH_SIZE), 4, 4),
            )

        self.create_caches()

    def create_caches(self):
        self.bboxes_cache = np.full((FRAME_BATCH_SIZE, p.getNumBodies(), 16), -1, dtype=np.float32)

        if SAVE_CAMERA:
            self.cameraV_cache = np.zeros((FRAME_BATCH_SIZE, 4, 4), dtype=np.float32)
            self.cameraP_cache = np.zeros((FRAME_BATCH_SIZE, 4, 4), dtype=np.float32)

    def write_to_file(self, env):
        frame_count = frame_to_entry_idx(env.simulator.frame_count)
        new_lines = frame_count % FRAME_BATCH_SIZE
        if new_lines == 0:
            return

        start_pos = frame_count - new_lines
        self.bboxes[start_pos:frame_count] = self.bboxes_cache[:new_lines]

        if SAVE_CAMERA:
            self.cameraV[start_pos:frame_count] = self.cameraV_cache[:new_lines]
            self.cameraP[start_pos:frame_count] = self.cameraP_cache[:new_lines]

        self.create_caches()

    def step_callback(self, env, _):
        if not is_subsampled_frame(env.simulator.frame_count):
            return

        # Clear debug drawings.
        if DEBUG_DRAW:
            p.removeAllUserDebugItems()

        renderer = env.simulator.renderer
        ins_seg = renderer.render_robot_cameras(modes="ins_seg")[0][:, :, 0]
        ins_seg = np.round(ins_seg * MAX_INSTANCE_COUNT).astype(int)
        id_seg = renderer.get_pb_ids_for_instance_ids(ins_seg)

        frame_idx = frame_to_entry_idx(env.simulator.frame_count) % FRAME_BATCH_SIZE
        filled_obj_idx = 0

        for body_id in np.unique(id_seg):
            if body_id == -1 or body_id not in env.simulator.scene.objects_by_id:
                continue

            # Get the object semantic class ID
            obj = env.scene.objects_by_id[body_id]
            if not isinstance(obj, URDFObject):
                # Ignore robots etc.
                continue

            class_id = env.simulator.class_name_to_class_id.get(obj.category, SemanticClass.SCENE_OBJS)

            # 2D bounding box
            this_object_pixels_positions = np.argwhere(id_seg == body_id)
            bb_top_left = np.min(this_object_pixels_positions, axis=0)
            bb_bottom_right = np.max(this_object_pixels_positions, axis=0)

            # 3D bounding box
            world_frame_center, world_frame_orientation, base_frame_extent, _ = obj.get_base_aligned_bounding_box(
                body_id=body_id, visual=True
            )
            world_frame_pose = np.concatenate([world_frame_center, world_frame_orientation])
            camera_frame_pose = env.simulator.renderer.transform_pose(world_frame_pose)
            camera_frame_center = camera_frame_pose[:3]
            camera_frame_orientation = camera_frame_pose[3:]

            # Debug-mode drawing of the bounding box.
            if DEBUG_DRAW and obj.category not in ("walls", "floors", "ceilings"):
                bbox_frame_vertex_positions = np.array(list(itertools.product((1, -1), repeat=3))) * (
                    base_frame_extent / 2
                )
                bbox_transform = utils.quat_pos_to_mat(world_frame_center, world_frame_orientation)
                world_frame_vertex_positions = trimesh.transformations.transform_points(
                    bbox_frame_vertex_positions, bbox_transform
                )
                for i, from_vertex in enumerate(world_frame_vertex_positions):
                    for j, to_vertex in enumerate(world_frame_vertex_positions):
                        if j <= i:
                            p.addUserDebugLine(from_vertex, to_vertex, [1.0, 0.0, 0.0], 1, 0)

            # Record the results.
            self.bboxes_cache[frame_idx, filled_obj_idx, 0] = body_id
            self.bboxes_cache[frame_idx, filled_obj_idx, 1] = class_id
            self.bboxes_cache[frame_idx, filled_obj_idx, 2:4] = bb_top_left
            self.bboxes_cache[frame_idx, filled_obj_idx, 4:6] = bb_bottom_right - bb_top_left
            self.bboxes_cache[frame_idx, filled_obj_idx, 6:9] = camera_frame_center
            self.bboxes_cache[frame_idx, filled_obj_idx, 9:13] = camera_frame_orientation
            self.bboxes_cache[frame_idx, filled_obj_idx, 13:16] = base_frame_extent
            filled_obj_idx += 1

        if SAVE_CAMERA:
            self.cameraV_cache[frame_idx] = renderer.V
            self.cameraP_cache[frame_idx] = renderer.P

        if frame_idx == FRAME_BATCH_SIZE - 1:
            self.write_to_file(env)

    def end_callback(self, env, _):
        self.write_to_file(env)


def main():
    args = parse_args()

    print(args)

    def get_imvotenet_callbacks(demo_name, out_dir):
        path = os.path.join(out_dir, demo_name + "_data.h5py")
        h5py_file = h5py.File(path, "w")
        extractors = [PointCloudExtractor(h5py_file), BBoxExtractor(h5py_file)]

        return (
            [extractor.start_callback for extractor in extractors],
            [extractor.step_callback for extractor in extractors],
            [extractor.end_callback for extractor in extractors] + [lambda a, b: h5py_file.close()],
            [],
        )

    # TODO: Set resolution to match model.
    behavior_demo_batch(
        args.demo_root,
        args.log_manifest,
        args.out_dir,
        get_imvotenet_callbacks,
        image_size=(480, 480),
        ignore_errors=True,
        debug_display=DEBUG_DRAW,
    )


if __name__ == "__main__":
    main()
