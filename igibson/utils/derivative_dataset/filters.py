import itertools
from dataclasses import dataclass

import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.utils.constants import MAX_CLASS_COUNT, MAX_INSTANCE_COUNT, SemanticClass
from igibson.utils.semantics_utils import CLASS_NAME_TO_CLASS_ID
from igibson.utils.transform_utils import bbox_in_img_frame

STRUCTURE_CLASSES = ["walls", "ceilings", "floors"]


@dataclass
class TooCloseFilter:
    min_dist: float = 0.0
    max_dist: float = float("inf")
    max_allowed_fraction_outside_threshold: float = 0.0

    def __call__(self, env, objs_of_interest):
        renderer: MeshRenderer = env.simulator.renderer
        depth_img = np.linalg.norm(renderer.render(modes=("3d"))[0], axis=-1)
        outside_range_pixels = np.count_nonzero(np.logical_or(depth_img < self.min_dist, depth_img > self.max_dist))
        return outside_range_pixels / len(depth_img.flatten()) <= self.max_allowed_fraction_outside_threshold


@dataclass
class TooMuchStructureFilter:
    max_allowed_fraction_of_structure: float

    def __call__(self, env, objs_of_interest):
        seg = env.simulator.renderer.render(modes=("seg"))[0][:, :, 0]
        seg_int = np.round(seg * MAX_CLASS_COUNT).astype(int).flatten()
        pixels_of_wall = np.count_nonzero(np.isin(seg_int, [CLASS_NAME_TO_CLASS_ID[x] for x in STRUCTURE_CLASSES]))
        return pixels_of_wall / len(seg_int) < self.max_allowed_fraction_of_structure


@dataclass
class TooMuchOfSameObjectFilter:
    threshold: float

    def __call__(self, env, objs_of_interest):
        seg, ins_seg = env.simulator.renderer.render(modes=("seg", "ins_seg"))

        # Get body ID per pixel
        ins_seg = np.round(ins_seg[:, :, 0] * MAX_INSTANCE_COUNT).astype(int)
        body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(ins_seg)

        # Use category to remove walls, floors, ceilings
        seg_int = np.round(seg[:, :, 0] * MAX_CLASS_COUNT).astype(int)
        pixels_of_wall = np.isin(seg_int, [CLASS_NAME_TO_CLASS_ID[x] for x in STRUCTURE_CLASSES])

        relevant_body_ids = body_ids[np.logical_not(pixels_of_wall)]

        return max(np.bincount(relevant_body_ids)) / len(body_ids.flatten()) < self.threshold


@dataclass
class NoRelevantObjectInFOVFilter:
    min_bbox_vertices_in_fov: int = 4

    def __call__(self, env, objs_of_interest):
        # Pick an object
        for obj in objs_of_interest:
            bbox_vertices = bbox_in_img_frame(obj, env.simulator.renderer)
            if len(bbox_vertices) >= self.min_bbox_vertices_in_fov:
                return True

        return False


@dataclass
class NoRelevantObjectInImageFilter:
    threshold: float = 0.2

    def __call__(self, env, objs_of_interest):
        seg = env.simulator.renderer.render(modes="ins_seg")[0][:, :, 0]
        seg = np.round(seg * MAX_INSTANCE_COUNT).astype(int)
        body_ids = env.simulator.renderer.get_pb_ids_for_instance_ids(seg)

        obj_body_ids = [x for obj in objs_of_interest for x in obj.get_body_ids()]
        relevant = np.count_nonzero(np.isin(body_ids, obj_body_ids))
        return relevant / len(seg.flatten()) > self.threshold

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


@dataclass
class PointInObjectFilter:
    def __call__(self, env, objs_of_interest):
        # Camera position
        cam_pos = env.simulator.renderer.camera
        target_pos = env.simulator.renderer.target
        target_dir = target_pos - cam_pos
        target_dir /= np.linalg.norm(target_dir)

        test_target = cam_pos + target_dir * 0.01
        if p.rayTest(cam_pos, test_target)[0][0] != -1:
            return False

        return True
