import inspect
import logging
import random
from enum import IntEnum
from math import ceil

import gym
import numpy as np
import pybullet as p
from scipy.spatial.transform import Rotation

from igibson import object_states
from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet
from igibson.external.pybullet_tools.utils import set_joint_position
from igibson.object_states.on_floor import RoomFloor
from igibson.object_states.utils import get_center_extent, sample_kinematics
from igibson.objects.articulated_object import URDFObject
from igibson.objects.object_base import BaseObject
from igibson.robots import BaseRobot, behavior_robot
from igibson.robots.behavior_robot import DEFAULT_BODY_OFFSET_FROM_FLOOR, BehaviorRobot
from igibson.tasks.behavior_task import BehaviorTask
from igibson.utils.behavior_robot_motion_planning_utils import (
    get_pose3d_hand_collision_fn,
    plan_base_motion_br,
    plan_hand_motion_br,
)
from igibson.utils.grasp_planning_utils import get_grasp_poses_for_object, get_grasp_position_for_open
from igibson.utils.utils import restoreState

logger = logging.getLogger(__name__)


class B1KActionPrimitive(IntEnum):
    NAVIGATE_TO = 0
    PICK = 1
    PLACE = 2
    TOGGLE = 3
    PULL = 4
    PUSH = 5


class B1KActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, task, scene, robot, arm=None):
        super().__init__(task, scene, robot)
        self.controller_functions = {
            B1KActionPrimitive.NAVIGATE_TO: self._navigate_to,
            B1KActionPrimitive.PICK: self._pick,
            B1KActionPrimitive.PLACE: self._place,
            B1KActionPrimitive.TOGGLE: self._toggle,
            B1KActionPrimitive.PULL: self._pull,
            B1KActionPrimitive.PUSH: self._push,
        }
        if arm is None:
            self.arm = self.robot.default_arm
            logger.warning("Grasping with the default arm: {}".format(arm))

    def get_action_space(self):
        if ACTIVITY_RELEVANT_OBJECTS_ONLY:
            assert isinstance(self.task, BehaviorTask), "Activity relevant objects can only be used for BEHAVIOR tasks."
            self.addressable_objects = [
                item
                for item in self.task.object_scope.values()
                if isinstance(item, URDFObject) or isinstance(item, RoomFloor)
            ]
        else:
            self.addressable_objects = set(self.scene.objects_by_name.values())
            if isinstance(self.task, BehaviorTask):
                self.addressable_objects.update(self.task.object_scope.values())
            self.addressable_objects = list(self.addressable_objects)

        # Filter out the robots.
        self.addressable_objects = [obj for obj in self.addressable_objects if not isinstance(obj, BaseRobot)]

        self.num_objects = len(self.addressable_objects)
        return gym.spaces.Tuple(
            [gym.spaces.Discrete(self.num_objects), gym.spaces.Discrete(len(StarterSemanticActionPrimitive))]
        )

    def get_action_from_primitive_and_object(self, primitive: StarterSemanticActionPrimitive, obj: BaseObject):
        assert obj in self.addressable_objects
        primitive_int = int(primitive)
        return primitive_int, self.addressable_objects.index(obj)

    def _get_obj_in_hand(self):
        obj_in_hand_id = self.robot._ag_obj_in_hand[self.arm]  # TODO(MP): Expose this interface.
        obj_in_hand = self.scene.objects_by_id[obj_in_hand_id] if obj_in_hand_id is not None else None
        return obj_in_hand

    def apply(self, action):
        # Decompose the tuple
        action_idx, obj_idx = action

        # Find the target object.
        target_obj = self.addressable_objects[obj_idx]

        # Find the appropriate action generator.
        action = StarterSemanticActionPrimitive(action_idx)
        return self.controller_functions[action](target_obj)

    def open(self, obj):
        yield from self._open_or_close(obj, True)

    def close(self, obj):
        yield from self._open_or_close(obj, False)

    def _open_or_close(self, obj, should_open):
        hand_collision_fn = get_pose3d_hand_collision_fn(
            self.robot, None, self._get_collision_body_ids(include_robot=True)
        )

        # Open the hand first
        yield from self._execute_release()

        # Don't do anything if the object is already closed and we're trying to close.
        if not should_open and not obj.states[object_states.Open].get_value():
            return

        grasp_data = get_grasp_position_for_open(self.robot, obj, should_open)
        if grasp_data is None:
            if should_open and obj.states[object_states.Open].get_value():
                # It's already open so we're good
                return
            else:
                # We were trying to do something but didn't have the data.
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Could not sample grasp position for object.",
                    {"object": obj},
                )

        grasp_pose, target_poses, object_direction, joint_info, grasp_required = grasp_data
        with UndoableContext(self.robot):
            if hand_collision_fn(grasp_pose):
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Rejecting grasp pose due to collision.",
                    {"object": obj, "grasp_pose": grasp_pose},
                )

        # Prepare data for the approach later.
        approach_pos = grasp_pose[0] + object_direction * OPEN_GRASP_APPROACH_DISTANCE
        approach_pose = (approach_pos, grasp_pose[1])

        # If the grasp pose is too far, navigate
        [bid] = obj.get_body_ids()  # TODO: Fix this!
        check_joint = (bid, joint_info)
        yield from self._navigate_if_needed(obj, pos_on_obj=approach_pos, check_joint=check_joint)
        yield from self._navigate_if_needed(obj, pos_on_obj=grasp_pose[0], check_joint=check_joint)

        yield from self._move_hand(grasp_pose)

        # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
        # It's okay if we can't go all the way because we run into the object.
        indented_print("Performing grasp approach for open.")

        try:
            yield from self._move_hand_direct(approach_pose, ignore_failure=True, stop_on_contact=True)
        except ActionPrimitiveError:
            # An error will be raised when contact fails. If this happens, let's retreat back to the grasp pose.
            yield from self._move_hand_direct(grasp_pose, ignore_failure=True)
            raise

        if grasp_required:
            try:
                yield from self._execute_grasp()
            except ActionPrimitiveError:
                # Retreat back to the grasp pose.
                yield from self._execute_release()
                yield from self._move_hand_direct(grasp_pose, ignore_failure=True)
                raise

        for target_pose in target_poses:
            yield from self._move_hand_direct(
                target_pose, ignore_failure=True, max_steps_for_hand_move=MAX_STEPS_FOR_HAND_MOVE_WHEN_OPENING
            )

        # Moving to target pose often fails. Let's get the hand to apply the correct actions for its current pos
        # This prevents the hand from jerking into its desired position when we do a release.
        yield from self._move_hand_direct(
            self.robot.eef_links[self.arm].get_position_orientation(), ignore_failure=True
        )

        yield from self._execute_release()
        yield from self._reset_hand()

        if obj.states[object_states.Open].get_value() == should_open:
            return

    def grasp(self, obj):
        # Don't do anything if the object is already grasped.
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand == obj:
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when hand is already full.",
                    {"object": obj, "object_in_hand": obj_in_hand},
                )

        hand_collision_fn = get_pose3d_hand_collision_fn(
            self.robot, None, self._get_collision_body_ids(include_robot=True)
        )
        if self._get_obj_in_hand() != obj:
            # Open the hand first
            yield from self._execute_release()

            # Allow grasping from suboptimal extents if we've tried enough times.
            force_allow_any_extent = np.random.rand() < 0.5
            grasp_poses = get_grasp_poses_for_object(self.robot, obj, force_allow_any_extent=force_allow_any_extent)
            grasp_pose, object_direction = random.choice(grasp_poses)
            with UndoableContext(self.robot):
                if hand_collision_fn(grasp_pose):
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.SAMPLING_ERROR,
                        "Rejecting grasp pose candidate due to collision",
                        {"grasp_pose": grasp_pose},  # TODO: Add more info about collision.
                    )

            # Prepare data for the approach later.
            approach_pos = grasp_pose[0] + object_direction * GRASP_APPROACH_DISTANCE
            approach_pose = (approach_pos, grasp_pose[1])

            # If the grasp pose is too far, navigate.
            yield from self._navigate_if_needed(obj, pos_on_obj=approach_pos)
            yield from self._navigate_if_needed(obj, pos_on_obj=grasp_pose[0])

            yield from self._move_hand(grasp_pose)

            # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
            # It's okay if we can't go all the way because we run into the object.
            indented_print("Performing grasp approach.")
            try:
                yield from self._move_hand_direct(approach_pose, ignore_failure=True, stop_on_contact=True)
            except ActionPrimitiveError:
                # An error will be raised when contact fails. If this happens, let's retry.
                # Retreat back to the grasp pose.
                yield from self._move_hand_direct(grasp_pose, ignore_failure=True)
                raise

            indented_print("Grasping.")
            try:
                yield from self._execute_grasp()
            except ActionPrimitiveError:
                # Retreat back to the grasp pose.
                yield from self._move_hand_direct(grasp_pose, ignore_failure=True)
                raise

        indented_print("Moving hand back to neutral position.")
        yield from self._reset_hand()

        if self._get_obj_in_hand() == obj:
            return

    def place_on_top(self, obj):
        yield from self._place_with_predicate(obj, object_states.OnTop)

    def place_inside(self, obj):
        yield from self._place_with_predicate(obj, object_states.Inside)

    def toggle_on(self, obj):
        yield from self._toggle(obj, True)

    def toggle_off(self, obj):
        yield from self._toggle(obj, False)

    def _toggle(self, obj, value):
        if obj.states[object_states.ToggledOn].get_value() == value:
            return

        # Put the hand in the toggle marker.
        toggle_state = obj.states[object_states.ToggledOn]
        toggle_position = toggle_state.get_link_position()
        yield from self._navigate_if_needed(obj, toggle_position)

        hand_orientation = self.robot.eef_links[self.arm].get_orientation()  # Just keep the current hand orientation.
        desired_hand_pose = (toggle_position, hand_orientation)

        yield from self._move_hand_direct(desired_hand_pose, ignore_failure=True)

        # Put hand back where it was.
        yield from self._reset_hand()

        if obj.states[object_states.ToggledOn].get_value() != value:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR, "Failed to toggle object.", {"object": object}
            )

    def _place_with_predicate(self, obj, predicate):
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "Cannot place object if not holding one."
            )

        obj_pose = self._sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj)
        hand_pose = self._get_hand_pose_for_object_pose(obj_pose)
        yield from self._navigate_if_needed(obj, pos_on_obj=hand_pose[0])
        yield from self._move_hand(hand_pose)
        yield from self._execute_release()
        yield from self._reset_hand()

        if obj_in_hand.states[predicate].get_value(obj):
            return

    def _move_hand(self, target_pose):
        target_pose_in_correct_format = list(target_pose[0]) + list(p.getEulerFromQuaternion(target_pose[1]))

        # Define the sampling domain.
        cur_pos = np.array(self.robot.get_position())
        target_pos = np.array(target_pose[0])
        both_pos = np.array([cur_pos, target_pos])
        min_pos = np.min(both_pos, axis=0) - HAND_SAMPLING_DOMAIN_PADDING
        max_pos = np.max(both_pos, axis=0) + HAND_SAMPLING_DOMAIN_PADDING

        with UndoableContext(self.robot):
            plan = plan_hand_motion_br(
                robot=self.robot,
                obj_in_hand=self._get_obj_in_hand(),
                end_conf=target_pose_in_correct_format,
                hand_limits=(min_pos, max_pos),
                obstacles=self._get_collision_body_ids(include_robot=True),
            )

        if plan is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "Could not make a hand motion plan.",
                {"target_pose": target_pose},
            )

        # Follow the plan to navigate.
        indented_print("Plan has %d steps.", len(plan))
        for i, xyz_rpy in enumerate(plan):
            indented_print("Executing grasp plan step %d/%d", i + 1, len(plan))
            pose = (xyz_rpy[:3], p.getQuaternionFromEuler(xyz_rpy[3:]))
            yield from self._move_hand_direct(pose)

    def _move_hand_direct(self, target_pose, **kwargs):
        yield from self._move_hand_direct_relative_to_robot(self._get_pose_in_robot_frame(target_pose), **kwargs)

    def _move_hand_direct_relative_to_robot(
        self,
        relative_target_pose,
        ignore_failure=False,
        max_steps_for_hand_move=MAX_STEPS_FOR_HAND_MOVE,
        stop_on_contact=False,
    ):
        for _ in range(max_steps_for_hand_move):
            if stop_on_contact and p.getContactPoints(self.robot.eef_links[self.arm].body_id):  # TODO(MP): Generalize
                return

            action = convert_behavior_robot_part_pose_to_action(self.robot, hand_target_pose=relative_target_pose)
            if action is None:
                if stop_on_contact:
                    raise ActionPrimitiveError(
                        ActionPrimitiveError.Reason.EXECUTION_ERROR,
                        "No contact was made when contact was required as a move post-condition.",
                    )
                return

            yield action

        if not ignore_failure:
            current_pose = self.robot.get_relative_eef_pose()
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Could not move gripper to desired position.",
                {"relative_target_pose": relative_target_pose, "relative_current_pose": current_pose},
            )

        if stop_on_contact:
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.EXECUTION_ERROR, "No contact was made.")

    def _execute_grasp(self):
        action = np.zeros(self.robot.action_dim)
        action[self.robot.controller_action_idx["gripper_right_hand"]] = -1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            yield action

        # Do nothing for a bit so that AG can trigger.
        for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
            yield np.zeros(self.robot.action_dim)

        if self._get_obj_in_hand() is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "No object detected in hand after executing grasp.",
            )

    def _execute_release(self):
        action = np.zeros(self.robot.action_dim)
        action[self.robot.controller_action_idx["gripper_right_hand"]] = 1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            # Otherwise, keep applying the action!
            yield action

        # Do nothing for a bit so that AG can trigger.
        for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
            yield np.zeros(self.robot.action_dim)

        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Object still detected in hand after executing release.",
                {"object_in_hand": self._get_obj_in_hand()},
            )

    def _reset_hand(self):
        default_pose = p.multiplyTransforms(
            # TODO(MP): Generalize.
            *self.robot.get_position_orientation(),
            *behavior_robot.RIGHT_HAND_LOC_POSE_TRACKED,
        )
        yield from self._move_hand(default_pose)

    def _navigate_to_pose(self, pose_2d):
        # TODO(lowprio-MP): Make this less awful.
        start_xy = np.array(self.robot.get_position())[:2]
        end_xy = pose_2d[:2]
        center_xy = (start_xy + end_xy) / 2
        circle_radius = np.linalg.norm(end_xy - start_xy)  # circle with 2x diameter as the distance

        def sample_fn():
            # With some probability, sample from a circle centered around the start & goal.
            within_circle = np.random.rand() > BIRRT_SAMPLING_CIRCLE_PROBABILITY

            while True:
                random_point = np.array(self.scene.get_random_point()[1][:2])

                if within_circle and np.linalg.norm(random_point - center_xy) > circle_radius:
                    continue

                return random_point[0], random_point[1], np.random.uniform(-np.pi, np.pi)

        with UndoableContext(self.robot):
            # Note that the plan returned by this planner only contains xy pairs & not yaw.
            plan = plan_base_motion_br(
                robot=self.robot,
                obj_in_hand=self._get_obj_in_hand(),
                end_conf=pose_2d,
                sample_fn=sample_fn,
                obstacles=self._get_collision_body_ids(),
            )

        if plan is None:
            # TODO: Would be great to produce a more informative error.
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.PLANNING_ERROR, "Could not make a navigation plan.")

        # Follow the plan to navigate.
        indented_print("Plan has %d steps.", len(plan))
        for i, pose_2d in enumerate(plan):
            indented_print("Executing navigation plan step %d/%d", i + 1, len(plan))
            low_precision = True if i < len(plan) - 1 else False
            yield from self._navigate_to_pose_direct(pose_2d, low_precision=low_precision)

        # Match the final desired yaw.
        # yield from self._rotate_in_place(pose_2d[2])

    def _rotate_in_place(self, yaw, low_precision=False):
        cur_pos = self.robot.get_position()
        target_pose = self._get_robot_pose_from_2d_pose((cur_pos[0], cur_pos[1], yaw))
        for _ in range(MAX_STEPS_FOR_WAYPOINT_NAVIGATION):
            action = convert_behavior_robot_part_pose_to_action(
                self.robot, body_target_pose=self._get_pose_in_robot_frame(target_pose), low_precision=low_precision
            )
            if action is None:
                indented_print("Rotate is complete.")
                break

            yield action

    def _navigate_if_needed(self, obj, pos_on_obj=None, **kwargs):
        if pos_on_obj is not None:
            if self._get_dist_from_point_to_shoulder(pos_on_obj) < HAND_DISTANCE_THRESHOLD:
                # No need to navigate.
                return
        elif obj.states[object_states.InReachOfRobot].get_value():
            return

        yield from self._navigate_to_obj(obj, pos_on_obj=pos_on_obj, **kwargs)

    def _navigate_to_obj(self, obj, pos_on_obj=None, **kwargs):
        if isinstance(obj, RoomFloor):
            # TODO(lowprio-MP): Pos-on-obj for the room navigation?
            pose = self._sample_pose_in_room(obj.room_instance)
        else:
            pose = self._sample_pose_near_object(obj, pos_on_obj=pos_on_obj, **kwargs)

        yield from self._navigate_to_pose(pose)

    def _navigate_to_pose_direct(self, pose_2d, low_precision=False):
        # First, rotate the robot to face towards the waypoint.
        # yield from self._rotate_in_place(pose_2d[2])

        # Keep the same orientation until the target.
        pose = self._get_robot_pose_from_2d_pose(pose_2d)
        for _ in range(MAX_STEPS_FOR_WAYPOINT_NAVIGATION):
            action = convert_behavior_robot_part_pose_to_action(
                self.robot, body_target_pose=self._get_pose_in_robot_frame(pose), low_precision=low_precision
            )
            if action is None:
                indented_print("Move is complete.")
                return

            yield action

        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.EXECUTION_ERROR,
            "Could not move robot to desired waypoint.",
            {"target_pose": pose, "current_pose": self.robot.get_position_orientation()},
        )

    def _sample_pose_near_object(self, obj, pos_on_obj=None, **kwargs):
        if pos_on_obj is None:
            pos_on_obj = self._sample_position_on_aabb_face(obj)

        pos_on_obj = np.array(pos_on_obj)
        obj_rooms = obj.in_rooms if obj.in_rooms else [self.scene.get_room_instance_by_point(pos_on_obj[:2])]
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
            distance = np.random.uniform(0.2, 1.0)
            yaw = np.random.uniform(-np.pi, np.pi)
            pose_2d = np.array(
                [pos_on_obj[0] + distance * np.cos(yaw), pos_on_obj[1] + distance * np.sin(yaw), yaw + np.pi]
            )

            # Check room
            if self.scene.get_room_instance_by_point(pose_2d[:2]) not in obj_rooms:
                indented_print("Candidate position is in the wrong room.")
                continue

            if not self._test_pose(pose_2d, pos_on_obj=pos_on_obj, **kwargs):
                continue

            return pose_2d

        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR, "Could not find valid position near object."
        )

    @staticmethod
    def _sample_position_on_aabb_face(target_obj):
        aabb_center, aabb_extent = get_center_extent(target_obj.states)
        # We want to sample only from the side-facing faces.
        face_normal_axis = random.choice([0, 1])
        face_normal_direction = random.choice([-1, 1])
        face_center = aabb_center + np.eye(3)[face_normal_axis] * aabb_extent * face_normal_direction
        face_lateral_axis = 0 if face_normal_axis == 1 else 1
        face_lateral_half_extent = np.eye(3)[face_lateral_axis] * aabb_extent / 2
        face_vertical_half_extent = np.eye(3)[2] * aabb_extent / 2
        face_min = face_center - face_vertical_half_extent - face_lateral_half_extent
        face_max = face_center + face_vertical_half_extent + face_lateral_half_extent
        return np.random.uniform(face_min, face_max)

    def _sample_pose_in_room(self, room: str):
        # TODO(MP): Bias the sampling near the agent.
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM):
            _, pos = self.scene.get_random_point_by_room_instance(room)
            yaw = np.random.uniform(-np.pi, np.pi)
            pose = (pos[0], pos[1], yaw)
            if self._test_pose(pose):
                return pose

        raise ActionPrimitiveError(
            ActionPrimitiveError.Reason.SAMPLING_ERROR, "Could not find valid position in room.", {"room": room}
        )

    def _sample_pose_with_object_and_predicate(self, predicate, held_obj, target_obj):
        with UndoableContext(self.robot):
            pred_map = {object_states.OnTop: "onTop", object_states.Inside: "inside"}
            result = sample_kinematics(
                pred_map[predicate],
                held_obj,
                target_obj,
                True,
                use_ray_casting_method=True,
                max_trials=MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE,
                skip_falling=True,
                z_offset=PREDICATE_SAMPLING_Z_OFFSET,
            )

            if not result:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.SAMPLING_ERROR,
                    "Could not sample position with object and predicate.",
                    {"target_object": target_obj, "held_object": held_obj, "predicate": pred_map[predicate]},
                )

            pos, orn = held_obj.get_position_orientation()
            return pos, orn

    @staticmethod
    def _detect_collision(body, obj_in_hand=None):
        collision = []
        obj_in_hand_id = None
        if obj_in_hand is not None:
            [obj_in_hand_id] = obj_in_hand.get_body_ids()
        for body_id in range(p.getNumBodies()):
            if body_id == body or body_id == obj_in_hand_id:
                continue
            closest_points = p.getClosestPoints(body, body_id, distance=0.01)
            if len(closest_points) > 0:
                collision.append(body_id)
                break
        return collision

    def _detect_robot_collision(self):
        # TODO(MP): Generalize.
        indented_print("Start collision test.")
        body = self._detect_collision(self.robot.links["body"].body_id)
        if body:
            indented_print("Body has collision with objects %s", body)
        left = self._detect_collision(self.robot.eef_links["left_hand"].body_id)
        if left:
            indented_print("Left hand has collision with objects %s", left)
        right = self._detect_collision(self.robot.eef_links[self.arm].body_id, self._get_obj_in_hand())
        if right:
            indented_print("Right hand has collision with objects %s", right)
        indented_print("End collision test.")

        return body or left or right

    def _test_pose(self, pose_2d, pos_on_obj=None, check_joint=None):
        with UndoableContext(self.robot):
            self.robot.set_position_orientation(*self._get_robot_pose_from_2d_pose(pose_2d))

            if pos_on_obj is not None:
                hand_distance = self._get_dist_from_point_to_shoulder(pos_on_obj)
                if hand_distance > HAND_DISTANCE_THRESHOLD:
                    indented_print("Candidate position failed shoulder distance test.")
                    return False

            if self._detect_robot_collision():
                indented_print("Candidate position failed collision test.")
                return False

            if check_joint is not None:
                body_id, joint_info = check_joint

                # Check at different positions of the joint.
                joint_range = joint_info.jointUpperLimit - joint_info.jointLowerLimit
                turn_steps = int(ceil(abs(joint_range) / JOINT_CHECKING_RESOLUTION))
                for i in range(turn_steps):
                    joint_pos = (i + 1) / turn_steps * joint_range + joint_info.jointLowerLimit
                    set_joint_position(body_id, joint_info.jointIndex, joint_pos)

                    if self._detect_robot_collision():
                        indented_print("Candidate position failed joint-move collision test.")
                        return False

            return True

    @staticmethod
    def _get_robot_pose_from_2d_pose(pose_2d):
        pos = np.array([pose_2d[0], pose_2d[1], DEFAULT_BODY_OFFSET_FROM_FLOOR])
        orn = p.getQuaternionFromEuler([0, 0, pose_2d[2]])
        return pos, orn

    def _get_pose_in_robot_frame(self, pose):
        body_pose = self.robot.get_position_orientation()
        world_to_body_frame = p.invertTransform(*body_pose)
        relative_target_pose = p.multiplyTransforms(*world_to_body_frame, *pose)
        return relative_target_pose

    def _get_collision_body_ids(self, include_robot=False):
        ids = []
        for obj in self.scene.get_objects():
            if not isinstance(obj, BaseRobot):
                ids.extend(obj.get_body_ids())

        if include_robot:
            # TODO(MP): Generalize
            ids.append(self.robot.eef_links["left_hand"].body_id)
            ids.append(self.robot.base_link.body_id)

        return ids

    def _get_dist_from_point_to_shoulder(self, pos):
        # TODO(MP): Generalize
        shoulder_pos_in_base_frame = np.array(
            self.robot.links["%s_shoulder" % self.arm].get_local_position_orientation()[0]
        )
        point_in_base_frame = np.array(self._get_pose_in_robot_frame((pos, [0, 0, 0, 1]))[0])
        shoulder_to_hand = point_in_base_frame - shoulder_pos_in_base_frame
        return np.linalg.norm(shoulder_to_hand)

    def _get_hand_pose_for_object_pose(self, desired_pose):
        obj_in_hand = self._get_obj_in_hand()

        assert obj_in_hand is not None

        # Get the object pose & the robot hand pose
        obj_in_world = obj_in_hand.get_position_orientation()
        hand_in_world = self.robot.eef_links[self.arm].get_position_orientation()

        # Get the hand pose relative to the obj pose
        world_in_obj = p.invertTransform(*obj_in_world)
        hand_in_obj = p.multiplyTransforms(*world_in_obj, *hand_in_world)

        # Now apply desired obj pose.
        desired_hand_pose = p.multiplyTransforms(*desired_pose, *hand_in_obj)

        return desired_hand_pose
