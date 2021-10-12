import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from igibson.robots import behavior_robot
from igibson.robots.behavior_robot import BehaviorRobot

DEFAULT_DIST_THRESHOLD = 0.02
DEFAULT_ANGLE_THRESHOLD = 0.05

BODY_KV = 0.5
BODY_KW = 0.2

LIMB_KV = 0.1
LIMB_KW = 0.15


def get_action_from_pose_to_pose(
    start_pose, end_pose, kv, kw, dist_threshold=DEFAULT_DIST_THRESHOLD, angle_threshold=DEFAULT_ANGLE_THRESHOLD
):
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
    return np.concatenate([diff_pos * kv, diff_angles * kw])


def get_action(robot: BehaviorRobot, body_target_pose=None, hand_target_pose=None, reset_others=False):
    assert body_target_pose is not None or hand_target_pose is not None

    # Compute the body information from the current frame.
    body = robot.parts["body"]
    body_pose = body.get_position_orientation()
    world_frame_to_body_frame = p.invertTransform(*body_pose)

    # Accumulate the actions in the correct order.
    action = np.zeros(26)

    # Compute the needed body motion
    if body_target_pose is not None:
        action[:6] = get_action_from_pose_to_pose(([0, 0, 0], [0, 0, 0, 1]), body_target_pose, BODY_KV, BODY_KW)

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
        action[19:25] = get_action_from_pose_to_pose(right_hand_pose_in_body_frame, hand_target_pose, LIMB_KV, LIMB_KW)
    else:
        # Move it back to the default position in with the below logic.
        parts_to_move_to_default_pos.append(("right_hand", 19, behavior_robot.RIGHT_HAND_LOC_POSE_TRACKED))

    # Move other parts to default positions.
    if reset_others:
        for part_name, start_idx, target_pose in parts_to_move_to_default_pos:
            part = robot.parts[part_name]
            part_pose_in_body_frame = p.multiplyTransforms(*world_frame_to_body_frame, *part.get_position_orientation())
            action[start_idx : start_idx + 6] = get_action_from_pose_to_pose(
                part_pose_in_body_frame, target_pose, LIMB_KV, LIMB_KW
            )

    # Return None if no action is needed.
    if np.all(action == 0):
        return None

    return action
