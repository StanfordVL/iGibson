import random

import cv2
import itertools
import numpy as np
import pybullet as p
from IPython import embed
from scipy.spatial.transform import Rotation as R
import trimesh

import igibson
from igibson.external.pybullet_tools.utils import (
    get_aabb,
    get_aabb_center,
    get_aabb_extent,
    get_link_pose,
    matrix_from_quat,
    stable_z_on_aabb,
)
from igibson.object_states.aabb import AABB
from igibson.object_states.object_state_base import CachingEnabledObjectState
from igibson.utils import sampling_utils, utils

_ON_TOP_RAY_CASTING_SAMPLING_PARAMS = {
    # "hit_to_plane_threshold": 0.1,  # TODO: Tune this parameter.
    "max_angle_with_z_axis": 0.17,
    "bimodal_stdev_fraction": 1e-6,
    "bimodal_mean_fraction": 1.0,
    "max_sampling_attempts": 50,
    "aabb_offset": 0.01,
}

_INSIDE_RAY_CASTING_SAMPLING_PARAMS = {
    # "hit_to_plane_threshold": 0.1,  # TODO: Tune this parameter.
    "max_angle_with_z_axis": 0.17,
    "bimodal_stdev_fraction": 0.4,
    "bimodal_mean_fraction": 0.5,
    "max_sampling_attempts": 100,
    "aabb_offset": -0.01,
}


def get_center_extent(obj_states):
    assert AABB in obj_states
    aabb = obj_states[AABB].get_value()
    center, extent = get_aabb_center(aabb), get_aabb_extent(aabb)
    return center, extent


def clear_cached_states(obj):
    for _, obj_state in obj.states.items():
        if isinstance(obj_state, CachingEnabledObjectState):
            obj_state.clear_cached_value()


def detect_collision(bodyA):
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id == bodyA:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def detect_collision(bodyA):
    collision = False
    for body_id in range(p.getNumBodies()):
        if body_id == bodyA:
            continue
        closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
        if len(closest_points) > 0:
            collision = True
            break
    return collision


def sample_kinematics(
    predicate,
    objA,
    objB,
    binary_state,
    use_ray_casting_method=False,
    max_trials=100,
    z_offset=0.05,
    skip_falling=False,
    allow_non_default_orientation=False,
):
    if not binary_state:
        raise NotImplementedError()

    sample_on_floor = predicate == "onFloor"

    if not use_ray_casting_method and not sample_on_floor and predicate not in objB.supporting_surfaces:
        return False

    objA.force_wakeup()
    if not sample_on_floor:
        objB.force_wakeup()

    state_id = p.saveState()
    for i in range(max_trials):
        pos = None
        if hasattr(objA, "orientations") and objA.orientations is not None:
            orientation = objA.sample_orientation()
        else:
            orientation = [0, 0, 0, 1]

        # Orientation needs to be set for stable_z_on_aabb to work correctly
        # Position needs to be set to be very far away because the object's
        # original position might be blocking rays (use_ray_casting_method=True)
        old_pos = np.array([0, 0, 1])
        objA.set_position_orientation(old_pos, orientation)

        if sample_on_floor:
            _, pos = objB.scene.get_random_point_by_room_instance(objB.room_instance)

            if pos is not None:
                pos[2] = stable_z_on_aabb(objA.get_body_id(), ([0, 0, pos[2]], [0, 0, pos[2]]))
        else:
            if use_ray_casting_method:
                if predicate == "onTop":
                    params = _ON_TOP_RAY_CASTING_SAMPLING_PARAMS
                elif predicate == "inside":
                    params = _INSIDE_RAY_CASTING_SAMPLING_PARAMS
                else:
                    assert False, "predicate is not onTop or inside: {}".format(predicate)

                aabb = get_aabb(objA.get_body_id())
                aabb_center, aabb_extent = get_aabb_center(aabb), get_aabb_extent(aabb)
                aabb_rotation = R.identity()

                bbox_center, bbox_orn, bbox_bf_extent, bbox_wf_extent = objA.get_base_aligned_bounding_box(visual=True)
                bbox_frame_vertex_positions = np.array(list(itertools.product((1, -1), repeat=3))) * (bbox_bf_extent / 2)
                bbox_transform = utils.quat_pos_to_mat(bbox_center, bbox_orn)
                world_frame_vertex_positions = trimesh.transformations.transform_points(
                    bbox_frame_vertex_positions, bbox_transform
                )
                objB_bounding_box = {
                    "bbox_center": bbox_center,
                    "bbox_orn": bbox_orn, 
                    "bbox_bf_extent": bbox_bf_extent, 
                    "bbox_wf_extent": bbox_wf_extent,
                    "bbox_frame_vertex_positions": bbox_frame_vertex_positions,
                    "world_frame_vertex_positions": world_frame_vertex_positions,
                }
                # for i, from_vertex in enumerate(world_frame_vertex_positions):
                #     for j, to_vertex in enumerate(world_frame_vertex_positions):
                #         if j <= i:
                #             p.addUserDebugLine(from_vertex, to_vertex, [1.0, 0.0, 0.0], 1, 0)
                # Rotate the AABB as needed.
                if allow_non_default_orientation:
                    # Rotate the AABB in one direction by 90 or -90 degrees.
                    axis = np.array(random.choice([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]]))
                    rotvec = axis * np.pi / 4  # Rotation vector's norm is the amount of rotation around axis
                    aabb_rotation = R.from_rotvec(rotvec)

                # Rotate the AABB
                aabb_extent = aabb_rotation.apply(aabb_extent)
                # import pdb; pdb.set_trace()

                # TODO: Get this to work with non-URDFObject objects.
                sampling_results = sampling_utils.sample_cuboid_on_object(
                    objB,
                    obj_bounding_box = objB_bounding_box,
                    num_samples=1,
                    cuboid_dimensions=aabb_extent,
                    axis_probabilities=[0, 0, 1],
                    refuse_downwards=True,
                    **params
                )

                sampled_vector = sampling_results[0][0]
                sampled_quaternion = sampling_results[0][2]

                sampling_success = sampled_vector is not None
                if sampling_success:
                    # Find the delta from the object's CoM to its AABB centroid
                    diff = old_pos - aabb_center

                    sample_rotation = R.from_quat(sampled_quaternion)
                    original_rotation = R.from_quat(orientation) * aabb_rotation
                    combined_rotation = sample_rotation * original_rotation

                    # Rotate it using the quaternion
                    rotated_diff = sample_rotation.apply(diff)

                    pos = sampled_vector + rotated_diff
                    orientation = combined_rotation.as_quat()
            else:
                random_idx = np.random.randint(len(objB.supporting_surfaces[predicate].keys()))
                body_id, link_id = list(objB.supporting_surfaces[predicate].keys())[random_idx]
                random_height_idx = np.random.randint(len(objB.supporting_surfaces[predicate][(body_id, link_id)]))
                height, height_map = objB.supporting_surfaces[predicate][(body_id, link_id)][random_height_idx]
                obj_half_size = np.max(objA.bounding_box) / 2 * 100
                obj_half_size_scaled = np.array([obj_half_size / objB.scale[1], obj_half_size / objB.scale[0]])
                obj_half_size_scaled = np.ceil(obj_half_size_scaled).astype(np.int)
                height_map_eroded = cv2.erode(height_map, np.ones(obj_half_size_scaled, np.uint8))

                valid_pos = np.array(height_map_eroded.nonzero())
                if valid_pos.shape[1] != 0:
                    random_pos_idx = np.random.randint(valid_pos.shape[1])
                    random_pos = valid_pos[:, random_pos_idx]
                    y_map, x_map = random_pos
                    y = y_map / 100.0 - 2
                    x = x_map / 100.0 - 2
                    z = height

                    pos = np.array([x, y, z])
                    pos *= objB.scale

                    # the supporting surface is defined w.r.t to the link frame, not
                    # the inertial frame
                    if link_id == -1:
                        link_pos, link_orn = p.getBasePositionAndOrientation(body_id)
                        dynamics_info = p.getDynamicsInfo(body_id, -1)
                        inertial_pos = dynamics_info[3]
                        inertial_orn = dynamics_info[4]
                        inv_inertial_pos, inv_inertial_orn = p.invertTransform(inertial_pos, inertial_orn)
                        link_pos, link_orn = p.multiplyTransforms(
                            link_pos, link_orn, inv_inertial_pos, inv_inertial_orn
                        )
                    else:
                        link_pos, link_orn = get_link_pose(body_id, link_id)
                    pos = matrix_from_quat(link_orn).dot(pos) + np.array(link_pos)
                    z = stable_z_on_aabb(objA.get_body_id(), ([0, 0, pos[2]], [0, 0, pos[2]]))
                    pos[2] = z

        if pos is None:
            success = False
        else:
            pos[2] += z_offset
            objA.set_position_orientation(pos, orientation)
            success = not detect_collision(objA.get_body_id())  # len(p.getContactPoints(objA.get_body_id())) == 0

        if igibson.debug_sampling:
            print("sample_kinematics", success)
            embed()

        if success:
            break
        else:
            p.restoreState(state_id)

    p.removeState(state_id)

    # if success and not skip_falling:
    #     objA.set_position_orientation(pos, orientation)
    #     # Let it fall for 0.2 second
    #     physics_timestep = p.getPhysicsEngineParameters()["fixedTimeStep"]
    #     for _ in range(int(0.2 / physics_timestep)):
    #         p.stepSimulation()
    #         if len(p.getContactPoints(bodyA=objA.get_body_id())) > 0:
    #             break

    return success
