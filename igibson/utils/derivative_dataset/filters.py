import collections
import itertools
import random

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT, SemanticClass
from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID

STRUCTURE_CLASSES = ["walls", "ceilings", "floors"]


def too_close_filter(min_dist=0, max_dist=float("inf"), max_allowed_fraction_outside_threshold=0):
    def filter_fn(env, objs_of_interest):
        renderer: MeshRenderer = env.simulator.renderer
        depth_img = np.linalg.norm(renderer.render(modes=("3d"))[0], axis=-1)
        outside_range_pixels = np.count_nonzero(np.logical_or(depth_img < min_dist, depth_img > max_dist))
        return outside_range_pixels / len(depth_img.flatten()) <= max_allowed_fraction_outside_threshold

    return filter_fn


def too_much_structure_filter(max_allowed_fraction_of_structure):
    def filter_fn(env, objs_of_interest):
        seg = env.simulator.renderer.render(modes=("seg"))[0][:, :, 0]
        seg_int = np.round(seg * MAX_CLASS_COUNT).astype(int).flatten()
        pixels_of_wall = np.count_nonzero(np.isin(seg_int, [CLASS_NAME_TO_CLASS_ID[x] for x in STRUCTURE_CLASSES]))
        return pixels_of_wall / len(seg_int) < max_allowed_fraction_of_structure

    return filter_fn


def too_much_of_same_object_in_fov_filter(threshold):
    def filter_fn(env, objs_of_interest):
        seg, ins_seg = env.simulator.renderer.render(modes=("seg", "ins_seg"))

        # Get body ID per pixel
        ins_seg = np.round(ins_seg[:, :, 0] * MAX_INSTANCE_COUNT).astype(int)
        body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)

        # Use category to remove walls, floors, ceilings
        seg_int = np.round(seg[:, :, 0] * MAX_CLASS_COUNT).astype(int)
        pixels_of_wall = np.isin(seg_int, [CLASS_NAME_TO_CLASS_ID[x] for x in STRUCTURE_CLASSES])

        relevant_body_ids = body_ids[np.logical_not(pixels_of_wall)]

        return max(np.bincount(relevant_body_ids)) / len(body_ids.flatten()) < threshold

    return filter_fn


def no_relevant_object_in_fov_filter(target_state, min_bbox_vertices_in_fov=4):
    def filter_fn(env, objs_of_interest):
        # Pick an object
        for obj in objs_of_interest:
            # Get the corners of the object's bbox
            (
                bbox_center_in_world,
                bbox_orn_in_world,
                bbox_extent_in_desired_frame,
                _,
            ) = obj.get_base_aligned_bounding_box(visual=True)
            bbox_rot = Rotation.from_quat(bbox_orn_in_world)
            bbox_unit_vertices = np.array(list(itertools.product((1, -1), repeat=3)))
            bbox_vertices = bbox_rot.apply(bbox_extent_in_desired_frame / 2 * bbox_unit_vertices)
            bbox_vertices_heterogeneous = np.concatenate([bbox_vertices.T, np.ones((1, len(bbox_vertices)))], axis=0)

            # Get the image coordinates of each vertex
            renderer: MeshRenderer = env.simulator.renderer
            bbox_vertices_in_camera_frame_heterogeneous = renderer.V @ bbox_vertices_heterogeneous
            bbox_vertices_in_camera_frame = (
                bbox_vertices_in_camera_frame_heterogeneous[:3] / bbox_vertices_in_camera_frame_heterogeneous[3:4]
            )
            projected_points_heterogeneous = renderer.get_intrinsics() @ bbox_vertices_in_camera_frame
            projected_points = projected_points_heterogeneous[:2] / projected_points_heterogeneous[2:3]

            points_valid = (
                np.all(projected_points >= 0, axis=0)
                & (projected_points[0] < renderer.width)
                & (projected_points[1] < renderer.height)
            )
            if np.count_nonzero(points_valid) >= min_bbox_vertices_in_fov:
                return True

        return False

    return filter_fn


def no_relevant_object_in_img_filter(target_state, threshold=0.2):
    def filter_fn(env, objs_of_interest):
        seg = env.simulator.renderer.render(modes="ins_seg")[0][:, :, 0]
        seg = np.round(seg * MAX_INSTANCE_COUNT).astype(int)
        body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(seg)

        obj_body_ids = [x for obj in objs_of_interest for x in obj.get_body_ids()]
        relevant = np.count_nonzero(np.isin(body_ids, obj_body_ids))
        return relevant / len(seg.flatten()) > threshold

        # Count how many pixels per object.
        # ctr = collections.Counter(body_ids.flatten())
        # if -1 in ctr:
        #     del ctr[-1]
        #
        # target_state_value = random.uniform(0, 1) < 0.5
        # target_pixel_count = 0
        # for body_id, pixel_count in ctr.items():
        #     obj = env.simulator.scene.objects_by_id[body_id]
        #     if target_state in obj.states:
        #         if obj.states[target_state].get_value() == target_state_value:
        #             target_pixel_count += pixel_count
        # return target_pixel_count / len(seg.flatten()) > threshold

    return filter_fn


def point_in_object_filter():
    def filter_fn(env, objs_of_interest):
        # Camera position
        cam_pos = env.simulator.renderer.camera
        target_pos = env.simulator.renderer.target
        target_dir = target_pos - cam_pos
        target_dir /= np.linalg.norm(target_dir)

        test_target = cam_pos + target_dir * 0.01
        if p.rayTest(cam_pos, test_target)[0][0] != -1:
            return False

        return True

    return filter_fn
