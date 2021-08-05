import argparse
import os

import numpy as np

import igibson
from igibson import object_states
from igibson.examples.behavior.behavior_demo_batch import behavior_demo_batch
from igibson.metrics.metric_base import MetricBase
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


class PointCloudExtractor(MetricBase):
    def __init__(self):
        self.points = []
        self.colors = []
        self.labels = []

    def step_callback(self, igbhvr_act_inst, log_reader):
        # TODO: Check how this compares to the outputs of SUNRGBD. This is a full sphere. Do we just want the robot FOV?
        w = igbhvr_act_inst.simulator.renderer.width
        h = igbhvr_act_inst.simulator.renderer.height

        pano_rgb = igbhvr_act_inst.simulator.renderer.get_equi(mode="rgb", use_robot_camera=True)
        pano_seg = igbhvr_act_inst.simulator.renderer.get_equi(mode="seg", use_robot_camera=True)
        pano_3d = igbhvr_act_inst.simulator.renderer.get_equi(mode="3d", use_robot_camera=True)
        depth = np.linalg.norm(pano_3d[:, :, :3], axis=2)
        theta = -np.arange(-np.pi / 2, np.pi / 2, np.pi / w)[:, None]
        phi = np.arange(-np.pi, np.pi, 2 * np.pi / (2 * h))[None, :]
        x = np.cos(theta) * np.sin(phi) * depth
        y = np.sin(theta) * depth
        z = np.cos(theta) * np.cos(phi) * depth

        mask = np.random.uniform(size=(w, 2 * h)) > 0.98

        self.points.append((np.stack([x[mask], y[mask], z[mask]]).T).astype(np.float32))
        self.points.append((pano_rgb[mask][:, :3]).astype(np.float32))
        self.points.append((pano_seg[mask][:, 0]).astype(np.int32))

    def gather_results(self):
        return {
            "point_cloud": {
                "points": self.points,
                "colors": self.colors,
                "labels": self.labels,
            }
        }


class BBox2DExtractor(MetricBase):
    def __init__(self):
        self.bboxes = []

    def step_callback(self, igbhvr_act_inst, log_reader):
        bbs = []
        renderer = igbhvr_act_inst.simulator.renderer
        seg = renderer.render_robot_cameras(modes="ins_seg")[0][:, :, 0]
        seg = np.round(seg * MAX_INSTANCE_COUNT)
        for obj in igbhvr_act_inst.simulator.scene.get_objects():
            # Check if the object is in sight.
            if not obj.states[object_states.InFOVOfRobot].get_value():
                continue

            # Get the coordinates of every position that matches this object in the current image.
            main_body_instances = [
                inst.id for inst in obj.renderer_instances if inst.pybullet_uuid == obj.get_body_id()
            ]
            this_object_pixels = np.isin(seg, main_body_instances)
            this_object_pixels_positions = np.argwhere(this_object_pixels)

            if len(this_object_pixels_positions) == 0:
                print("Skipping", obj.name)
                continue

            bb_top_left = np.min(this_object_pixels_positions, axis=0)
            bb_bottom_right = np.max(this_object_pixels_positions, axis=0)

            # Convert to the same semantic class ID
            class_id = igbhvr_act_inst.simulator.class_name_to_class_id.get(obj.category, SemanticClass.SCENE_OBJS)
            bbs.append((class_id, bb_top_left, bb_bottom_right - bb_top_left))

        # Add this frame's results to the overall results.
        self.bboxes.append(bbs)

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

    def gather_results(self):
        return {"bbox_2d": self.bboxes}


class BBox3DExtractor(MetricBase):
    def __init__(self):
        self.bboxes = []

    def step_callback(self, igbhvr_act_inst, log_reader):
        bbs = []
        renderer = igbhvr_act_inst.simulator.renderer
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
            transformed_pose = renderer.transform_pose(pose)
            center_cam = transformed_pose[:3]
            orientation_cam = transformed_pose[3:]

            # Convert to the same semantic class ID
            class_id = igbhvr_act_inst.simulator.class_name_to_class_id.get(obj.category, SemanticClass.SCENE_OBJS)
            bbs.append((class_id, center_cam, orientation_cam, extent))

        self.bboxes.append(bbs)

    def gather_results(self):
        return {"bbox_3d": self.bboxes}


def main():
    args = parse_args()

    def get_imvotenet_callbacks():
        extractors = [PointCloudExtractor(), BBox3DExtractor(), BBox2DExtractor()]

        return (
            [extractor.start_callback for extractor in extractors],
            [extractor.step_callback for extractor in extractors],
            [extractor.end_callback for extractor in extractors],
            [extractor.gather_results for extractor in extractors],
        )

    # TODO: Set resolution to match model.
    behavior_demo_batch(args.demo_root, args.log_manifest, args.out_dir, get_imvotenet_callbacks, image_size=(720, 720))


if __name__ == "__main__":
    main()
