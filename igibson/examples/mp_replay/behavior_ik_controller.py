import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from igibson.robots import behavior_robot
from igibson.robots.behavior_robot import BehaviorRobot

DEFAULT_DIST_THRESHOLD = 0.01
DEFAULT_ANGLE_THRESHOLD = 0.05

LOW_PRECISION_DIST_THRESHOLD = 0.1
LOW_PRECISION_ANGLE_THRESHOLD = 0.2

BODY_KV = 0.5
BODY_KW = 0.5
BODY_MAX_V = 0.3
BODY_MAX_W = 1

LIMB_KV = 0.15
LIMB_KW = 0.2
LIMB_MAX_V = 0.05
LIMB_MAX_W = 0.25


def get_action_from_pose_to_pose(start_pose, end_pose, kv, kw, max_v, max_w, dist_threshold, angle_threshold):
    start_pos, start_orn = start_pose
    start_rot = Rotation.from_quat(start_orn)

    end_pos, end_orn = end_pose
    end_rot = Rotation.from_quat(end_orn)

    diff_rot = end_rot * start_rot.inv()
    # Use pb to avoid the weirdness around intrinsic / extrinsic rotations
    diff_angles = np.array(p.getEulerFromQuaternion(diff_rot.as_quat()))

    diff_pos = np.array(end_pos) - np.array(start_pos)

    # Check if there's anything we need to do
    # TODO(replayMP): Stop pushing an action if we're unable to move for some reason.
    if diff_rot.magnitude() < angle_threshold and np.linalg.norm(diff_pos) < dist_threshold:
        return np.zeros(6)

    # Return the position error scaled by the gains
    return np.concatenate([np.clip(diff_pos * kv, -max_v, max_v), np.clip(diff_angles * kw, -max_w, max_w)])


def get_action(
    robot: BehaviorRobot, body_target_pose=None, hand_target_pose=None, reset_others=True, low_precision=False
):
    assert body_target_pose is not None or hand_target_pose is not None

    # Compute the body information from the current frame.
    body = robot.parts["body"]
    body_pose = body.get_position_orientation()
    world_frame_to_body_frame = p.invertTransform(*body_pose)

    # Accumulate the actions in the correct order.
    action = np.zeros(26)

    dist_threshold = LOW_PRECISION_DIST_THRESHOLD if low_precision else DEFAULT_DIST_THRESHOLD
    angle_threshold = LOW_PRECISION_ANGLE_THRESHOLD if low_precision else DEFAULT_ANGLE_THRESHOLD

    # Compute the needed body motion
    if body_target_pose is not None:
        # TODO(replayMP): Do we need to disregard upwards movement?
        # body_target_pose_without_upwards = ((body_target_pose[0][0], body_target_pose[0][1], 0), body_target_pose[1])
        action[:6] = get_action_from_pose_to_pose(
            ([0, 0, 0], [0, 0, 0, 1]),
            body_target_pose,
            BODY_KV,
            BODY_KW,
            BODY_MAX_V,
            BODY_MAX_W,
            dist_threshold,
            angle_threshold,
        )

    # Keep a list of parts we'll move to default positions later. This is in correct order.
    parts_to_move_to_default_pos = [
        ("eye", 6, behavior_robot.EYE_LOC_POSE_TRACKED),
        ("left_hand", 12, behavior_robot.LEFT_HAND_LOC_POSE_TRACKED),
    ]

    # Take care of the right hand now.
    if hand_target_pose is not None:
        # Compute the needed right hand action
        right_hand = robot.parts["right_hand"]
        right_hand_pose_in_body_frame = p.multiplyTransforms(
            *world_frame_to_body_frame, *right_hand.get_position_orientation()
        )
        action[19:25] = get_action_from_pose_to_pose(
            right_hand_pose_in_body_frame,
            hand_target_pose,
            LIMB_KV,
            LIMB_KW,
            LIMB_MAX_V,
            LIMB_MAX_W,
            dist_threshold,
            angle_threshold,
        )
    else:
        # Move it back to the default position in with the below logic.
        parts_to_move_to_default_pos.append(("right_hand", 19, behavior_robot.RIGHT_HAND_LOC_POSE_TRACKED))

    # Move other parts to default positions.
    if reset_others:
        for part_name, start_idx, target_pose in parts_to_move_to_default_pos:
            part = robot.parts[part_name]
            part_pose_in_body_frame = p.multiplyTransforms(*world_frame_to_body_frame, *part.get_position_orientation())
            action[start_idx : start_idx + 6] = get_action_from_pose_to_pose(
                part_pose_in_body_frame,
                target_pose,
                LIMB_KV,
                LIMB_KW,
                LIMB_MAX_V,
                LIMB_MAX_W,
                dist_threshold,
                angle_threshold,
            )

    # Return None if no action is needed.
    if np.all(action == 0):
        return None

    return action
