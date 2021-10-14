import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from igibson.object_states.open import get_relevant_joints
from igibson.objects.articulated_object import URDFObject

PRISMATIC_JOINT_DIST_ACROSS_LATERAL_AXIS = 0.4
GUARANTEED_GRASPABLE_WIDTH = 0.1
GRASP_OFFSET = np.array([0, 0.09, -0.07])  # 9cm back and 7cm up.
GRASP_ANGLE = 0.52  # 30 degrees
NUM_LATERAL_SAMPLES = 10
LATERAL_MARGIN = 0.05


def get_grasp_position_for_open(robot, target_obj, should_open):
    grasp_candidates = []

    # Pick a moving link of the object.
    relevant_joints = get_relevant_joints(target_obj)
    for relevant_joint_info in relevant_joints:
        if relevant_joint_info.jointType == p.JOINT_REVOLUTE:
            per_joint_candidates = grasp_position_for_open_on_revolute_joint(
                robot, target_obj, relevant_joint_info, should_open
            )
        elif relevant_joint_info.jointType == p.JOINT_PRISMATIC:
            per_joint_candidates = grasp_position_for_open_on_prismatic_joint(
                robot, target_obj, relevant_joint_info, should_open
            )
        else:
            raise ValueError("Unknown joint type encountered while generating joint position.")
        grasp_candidates.extend(per_joint_candidates)


def grasp_position_for_open_on_prismatic_joint(robot, target_obj, target_joint_info, should_open):
    grasp_candidates = []

    # Pick a moving link of the object.
    relevant_joints = get_relevant_joints(target_obj)
    for relevant_joint_info in relevant_joints:
        link_id = relevant_joint_info.jointIndex

        # Get the bounding box of the child link.
        (
            bbox_center_in_world,
            bbox_quat_in_world,
            bbox_extent_in_link_frame,
            _,
        ) = target_obj.get_base_aligned_bounding_box(link_id=link_id, visual=False)

        # Get the part of the object away from the joint position/axis.
        # The link origin is where the joint is. Let's get the position of the origin w.r.t the CoM.
        dynamics_info = p.getDynamicsInfo(target_obj.get_body_id(), link_id)
        origin_pos, origin_orn = p.invertTransform(dynamics_info[3], dynamics_info[4])

        joint_axis = np.array(relevant_joint_info.jointAxis)
        origin_towards_com = -np.array(origin_pos)
        push_direction = np.cross(joint_axis, origin_towards_com)
        push_direction /= np.linalg.norm(push_direction)
        lateral_axis = np.cross(push_direction, joint_axis)

        # Match the axes to the canonical axes of the link bb.
        lateral_axis_idx = np.argmax(np.abs(lateral_axis))
        push_axis_idx = np.argmax(np.abs(push_direction))
        joint_axis_idx = np.argmax(np.abs(joint_axis))
        assert lateral_axis_idx != push_axis_idx
        assert lateral_axis_idx != joint_axis_idx
        assert push_axis_idx != joint_axis_idx

        # Find the correct side of the push axis the user is on.
        canonical_push_direction = np.eye(3)[push_axis_idx] * bbox_extent_in_link_frame[push_axis_idx] / 2
        points_along_push_axis = np.array([canonical_push_direction, -canonical_push_direction])
        push_axis_closer_idx, closer_surface_along_push_axis, _ = get_closest_point_to_point_in_world_frame(
            points_along_push_axis, (origin_pos, origin_orn), robot.get_position()
        )
        push_axis_closer_sign = -1 if push_axis_closer_idx == 1 else 1

        # Find the correct side of the lateral axis & go some distance along that direction.
        canonical_lateral_axis = np.eye(3)[lateral_axis_idx] * np.sign(origin_towards_com[lateral_axis_idx])
        lateral_position = (
            canonical_lateral_axis
            * PRISMATIC_JOINT_DIST_ACROSS_LATERAL_AXIS
            * bbox_extent_in_link_frame[lateral_axis_idx]
        )
        push_position = closer_surface_along_push_axis + lateral_position

        # Get the appropriate rotation
        palm = np.eye(3)[push_axis_idx] * -push_axis_closer_sign
        wrist = np.eye(3)[joint_axis_idx] * -np.sign(joint_axis[joint_axis_idx])
        lateral = -canonical_lateral_axis
        flat_rot = get_hand_rotation_from_axes(lateral, wrist, palm)

        # Finally apply our predetermined rotation around the X axis.
        grasp_orn_in_bbox_frame = flat_rot * Rotation.from_euler("X", -GRASP_ANGLE)
        push_quat = grasp_orn_in_bbox_frame.as_quat()

        # Now apply the grasp offset.
        grasp_pose = p.multiplyTransforms(bbox_center_in_world, bbox_quat_in_world, push_position, push_quat)
        offset_grasp_pose = p.multiplyTransforms(*grasp_pose, GRASP_OFFSET, [0, 0, 0, 1])
        grasp_candidates.append(offset_grasp_pose)

    return grasp_candidates


def grasp_position_for_open_on_revolute_joint(robot, target_obj, target_joint_info, should_open):
    grasp_candidates = []

    # Pick a moving link of the object.
    relevant_joints = get_relevant_joints(target_obj)
    for relevant_joint_info in relevant_joints:
        link_id = relevant_joint_info.jointIndex

        # Get the bounding box of the child link.
        (
            bbox_center_in_world,
            bbox_quat_in_world,
            bbox_extent_in_link_frame,
            _,
        ) = target_obj.get_base_aligned_bounding_box(link_id=link_id, visual=False)

        # Get the part of the object away from the joint position/axis.
        # The link origin is where the joint is. Let's get the position of the origin w.r.t the CoM.
        dynamics_info = p.getDynamicsInfo(target_obj.get_body_id(), link_id)
        origin_pos, origin_orn = p.invertTransform(dynamics_info[3], dynamics_info[4])

        joint_axis = np.array(relevant_joint_info.jointAxis)
        origin_towards_com = -np.array(origin_pos)
        push_direction = np.cross(joint_axis, origin_towards_com)
        push_direction /= np.linalg.norm(push_direction)
        lateral_axis = np.cross(push_direction, joint_axis)

        # Match the axes to the canonical axes of the link bb.
        lateral_axis_idx = np.argmax(np.abs(lateral_axis))
        push_axis_idx = np.argmax(np.abs(push_direction))
        joint_axis_idx = np.argmax(np.abs(joint_axis))
        assert lateral_axis_idx != push_axis_idx
        assert lateral_axis_idx != joint_axis_idx
        assert push_axis_idx != joint_axis_idx

        # Find the correct side of the push axis the user is on.
        canonical_push_direction = np.eye(3)[push_axis_idx] * bbox_extent_in_link_frame[push_axis_idx] / 2
        points_along_push_axis = np.array([canonical_push_direction, -canonical_push_direction])
        push_axis_closer_idx, closer_surface_along_push_axis, _ = get_closest_point_to_point_in_world_frame(
            points_along_push_axis, (origin_pos, origin_orn), robot.get_position()
        )
        push_axis_closer_sign = -1 if push_axis_closer_idx == 1 else 1

        # Find the correct side of the lateral axis & go some distance along that direction.
        canonical_lateral_axis = np.eye(3)[lateral_axis_idx] * np.sign(origin_towards_com[lateral_axis_idx])
        lateral_position = (
            canonical_lateral_axis
            * PRISMATIC_JOINT_DIST_ACROSS_LATERAL_AXIS
            * bbox_extent_in_link_frame[lateral_axis_idx]
        )
        push_position = closer_surface_along_push_axis + lateral_position

        # Get the appropriate rotation
        palm = np.eye(3)[push_axis_idx] * -push_axis_closer_sign
        wrist = np.eye(3)[joint_axis_idx] * -np.sign(joint_axis[joint_axis_idx])
        lateral = -canonical_lateral_axis
        flat_rot = get_hand_rotation_from_axes(lateral, wrist, palm)

        # Finally apply our predetermined rotation around the X axis.
        grasp_orn_in_bbox_frame = flat_rot * Rotation.from_euler("X", -GRASP_ANGLE)
        push_quat = grasp_orn_in_bbox_frame.as_quat()

        # Now apply the grasp offset.
        grasp_pose = p.multiplyTransforms(bbox_center_in_world, bbox_quat_in_world, push_position, push_quat)
        offset_grasp_pose = p.multiplyTransforms(*grasp_pose, GRASP_OFFSET, [0, 0, 0, 1])
        grasp_candidates.append(offset_grasp_pose)

    return grasp_candidates


def get_grasp_poses_for_object(robot, target_obj: URDFObject, force_allow_any_extent=True):
    # Grab the object's bounding box and attempt a top-down grasp.
    bbox_center_in_world, bbox_quat_in_world, bbox_extent_in_base_frame, _ = target_obj.get_base_aligned_bounding_box(
        visual=False
    )

    allow_any_extent = True
    if not force_allow_any_extent and np.any(bbox_extent_in_base_frame < GUARANTEED_GRASPABLE_WIDTH):
        allow_any_extent = False

    grasp_candidates = []
    all_possible_idxes = set(range(3))
    for pinched_axis_idx in all_possible_idxes:
        # If we're only accepting correctly-sized sides, let's check that.
        if not allow_any_extent and bbox_extent_in_base_frame[pinched_axis_idx] > GUARANTEED_GRASPABLE_WIDTH:
            continue

        # For this side, grab all the possible candidate positions (the other axes).
        other_axis_idxes = all_possible_idxes - {pinched_axis_idx}

        # For each axis we can go:
        for palm_normal_axis_idx in other_axis_idxes:
            positive_option_for_palm_normal = np.eye(3)[palm_normal_axis_idx]
            both_options_for_palm_normal = [positive_option_for_palm_normal, -positive_option_for_palm_normal]

            possible_grasp_centers_in_world = np.array(
                [
                    p.multiplyTransforms(
                        bbox_center_in_world,
                        bbox_quat_in_world,
                        possible_side * bbox_extent_in_base_frame / 2,
                        [0, 0, 0, 1],
                    )[0]
                    for possible_side in both_options_for_palm_normal
                ]
            )

            grasp_center_distances_to_robot = np.linalg.norm(
                possible_grasp_centers_in_world - np.array(robot.get_position())[None, :], axis=1
            )
            closer_option_idx = np.argmin(grasp_center_distances_to_robot)
            grasp_center_pos_in_bbox_frame = both_options_for_palm_normal[closer_option_idx]
            grasp_center_pos = possible_grasp_centers_in_world[closer_option_idx]
            towards_object_in_world_frame = bbox_center_in_world - grasp_center_pos
            towards_object_in_world_frame /= np.linalg.norm(towards_object_in_world_frame)

            # For the rotation, we want to get an orientation that points the palm towards the object.
            # The palm normally points downwards in its default position. We fit a rotation to this transform.
            palm_normal_in_bbox_frame = -grasp_center_pos_in_bbox_frame

            # We want the hand's backwards axis, e.g. the wrist, (which is Y+) to face closer to the robot.
            possible_backward_vectors = []
            for facing_side in [-1, 1]:
                possible_backward_vectors.append(np.eye(3)[pinched_axis_idx] * facing_side)

            possible_backward_vectors_in_world = np.array(
                [
                    p.multiplyTransforms(grasp_center_pos, bbox_quat_in_world, v, [0, 0, 0, 1])[0]
                    for v in possible_backward_vectors
                ]
            )

            backward_vector_distances_to_robot = np.linalg.norm(
                possible_backward_vectors_in_world - np.array(robot.get_position())[None, :], axis=1
            )
            best_wrist_side_in_bbox_frame = possible_backward_vectors[np.argmin(backward_vector_distances_to_robot)]

            # Fit a rotation to the vectors.
            # Something is wrong with the typical scipy approach so I rolled my own.
            # flat_grasp_rot, _ = Rotation.align_vectors([palm_normal_in_hand_frame, palm_forward], [palm_normal_in_bbox_frame, desired_forward])
            lateral_in_bbox_frame = np.cross(best_wrist_side_in_bbox_frame, palm_normal_in_bbox_frame)
            flat_grasp_rot = get_hand_rotation_from_axes(
                lateral_in_bbox_frame, best_wrist_side_in_bbox_frame, palm_normal_in_bbox_frame
            )

            # Finally apply our predetermined rotation around the X axis.
            bbox_orn_in_world = Rotation.from_quat(bbox_quat_in_world)
            grasp_orn_in_bbox_frame = flat_grasp_rot * Rotation.from_euler("X", -GRASP_ANGLE)
            grasp_orn_in_world_frame = bbox_orn_in_world * grasp_orn_in_bbox_frame
            grasp_quat = grasp_orn_in_world_frame.as_quat()

            # Now apply the grasp offset.
            grasp_pos = grasp_center_pos + grasp_orn_in_world_frame.apply(GRASP_OFFSET)

            # Append the center grasp to our list of candidates.
            grasp_pose = (grasp_pos, grasp_quat)
            grasp_candidates.append((grasp_pose, towards_object_in_world_frame))

            # Finally, we want to sample some different grasp candidates that involve lateral movements from
            # the center grasp.
            # First, compute the range of values we can sample from.
            lateral_axis_idx = (other_axis_idxes - {palm_normal_axis_idx}).pop()
            half_extent_in_lateral_direction = bbox_extent_in_base_frame[lateral_axis_idx] / 2
            lateral_range = half_extent_in_lateral_direction - LATERAL_MARGIN
            if lateral_range <= 0:
                continue

            # If there is a valid range, grab some samples from there.
            lateral_distance_samples = np.random.uniform(-lateral_range, lateral_range, NUM_LATERAL_SAMPLES)
            lateral_in_world_frame = bbox_orn_in_world.apply(lateral_in_bbox_frame)
            for lateral_distance in lateral_distance_samples:
                offset_grasp_pos = grasp_pos + lateral_distance * lateral_in_world_frame
                offset_grasp_pose = (offset_grasp_pos, grasp_quat)
                grasp_candidates.append((offset_grasp_pose, towards_object_in_world_frame))

    return grasp_candidates


def get_hand_rotation_from_axes(lateral, wrist, palm):
    rotmat = np.array([lateral, wrist, palm]).T
    assert np.isclose(np.dot(rotmat[:, 0], rotmat[:, 1]), 0)
    assert np.isclose(np.dot(rotmat[:, 0], rotmat[:, 2]), 0)
    assert np.isclose(np.dot(rotmat[:, 2], rotmat[:, 1]), 0)
    assert np.isclose(np.linalg.det(rotmat), 1)
    return Rotation.from_matrix(rotmat)


def get_closest_point_to_point_in_world_frame(
    vectors_in_arbitrary_frame, arbitrary_frame_to_world_frame, point_in_world
):
    vectors_in_world = np.array(
        [
            p.multiplyTransforms(*arbitrary_frame_to_world_frame, vector, [0, 0, 0, 1])[0]
            for vector in vectors_in_arbitrary_frame
        ]
    )

    vector_distances_to_point = np.linalg.norm(vectors_in_world - np.array(point_in_world)[None, :], axis=1)
    closer_option_idx = np.argmin(vector_distances_to_point)
    vector_in_arbitrary_frame = vectors_in_arbitrary_frame[closer_option_idx]
    vector_in_world_frame = vectors_in_world[closer_option_idx]

    return closer_option_idx, vector_in_arbitrary_frame, vector_in_world_frame
