import random

import numpy as np
import pybullet as p

from igibson import object_states
from igibson.examples.mp_replay import behavior_ik_controller
from igibson.examples.mp_replay.behavior_grasp_planning_utils import (
    get_grasp_poses_for_object,
    get_grasp_position_for_open,
)
from igibson.examples.mp_replay.behavior_motion_planning_utils import (
    get_pose3d_hand_collision_fn,
    plan_base_motion_br,
    plan_hand_motion_br,
)
from igibson.external.pybullet_tools.utils import get_center_extent
from igibson.object_states.on_floor import RoomFloor
from igibson.object_states.utils import sample_kinematics
from igibson.objects.articulated_object import URDFObject
from igibson.robots import behavior_robot
from igibson.robots.behavior_robot import BODY_OFFSET_FROM_FLOOR, BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene

HAND_DISTANCE_THRESHOLD = 0.9 * behavior_robot.HAND_DISTANCE_THRESHOLD

MAX_STEPS_FOR_HAND_MOVE = 100
MAX_STEPS_FOR_GRASP_OR_RELEASE = 30
MAX_WAIT_FOR_GRASP_OR_RELEASE = 10
MAX_STEPS_FOR_WAYPOINT_NAVIGATION = 600

MAX_ATTEMPTS_FOR_GRASPING = 100
MAX_ATTEMPTS_FOR_OPENING = 5
MAX_ATTEMPTS_FOR_OBJECT_NAVIGATION = 20

MAX_ATTEMPTS_FOR_SAMPLING_POSE_WITH_OBJECT_AND_PREDICATE = 20
MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT = 100
MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM = 60

BIRRT_SAMPLING_CIRCLE_PROBABILITY = 0.5
HAND_SAMPLING_DOMAIN_PADDING = 1  # Allow 1m of freedom around the sampling range.

GRASP_APPROACH_DISTANCE = 0.05

RIGHT_HAND_OBJECT_CARRYING_POSE = ([0.2, -0.12, -0.05], [-0.7, 0.7, 0.0, 0.15])


class MotionPrimitiveError(ValueError):
    pass


class UndoableContext(object):
    def __init__(self, robot: BehaviorRobot):
        self.robot = robot

    def __enter__(self):
        self.robot_data = self.robot.dump_state()
        self.state = p.saveState()

    def __exit__(self, *args):
        self.robot.load_state(self.robot_data)
        p.restoreState(self.state)
        p.removeState(self.state)


class MotionPrimitiveController(object):
    def __init__(self, scene, robot):
        self.scene: InteractiveIndoorScene = scene
        self.robot: BehaviorRobot = robot

    def _get_obj_in_hand(self):
        obj_in_hand_id = self.robot.parts["right_hand"].object_in_hand  # TODO(lowprio-replayMP): Generalize
        obj_in_hand = self.scene.objects_by_id[obj_in_hand_id] if obj_in_hand_id is not None else None
        return obj_in_hand

    def open(self, obj):
        yield from self._open_or_close(obj, True)

    def close(self, obj):
        yield from self._open_or_close(obj, False)

    def _open_or_close(self, obj, should_open):
        # Don't do anything if the object is already open.
        if obj.states[object_states.Open].get_value() == should_open:
            return

        for _ in range(MAX_ATTEMPTS_FOR_OPENING):
            try:
                grasp_pose, target_pose = get_grasp_position_for_open(self.robot, obj, should_open)
                # TODO(replayMP): How do we navigate to the correct side of a door?
                yield from self._navigate_if_needed(obj, pos_on_obj=grasp_pose[0])
                yield from self._move_hand(grasp_pose)
                yield from self._execute_grasp()
                yield from self._move_hand_direct(target_pose)
                yield from self._execute_release()

                if obj.states[object_states.Open].get_value() == should_open:
                    return
            except MotionPrimitiveError as e:
                print("Retrying open/close:", e)

        raise MotionPrimitiveError("Object could not be opened.")

    def grasp(self, obj):
        # Don't do anything if the object is already grasped.
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is not None:
            if obj_in_hand == obj:
                return
            else:
                raise MotionPrimitiveError("Cannot grasp when hand is already full.")

        hand_collision_fn = get_pose3d_hand_collision_fn(
            self.robot, None, self._get_collision_body_ids(include_robot=True)
        )
        for i in range(MAX_ATTEMPTS_FOR_GRASPING):
            try:
                if self._get_obj_in_hand() != obj:
                    # Open the hand first
                    yield from self._execute_release()

                    # Allow grasping from suboptimal extents if we've tried enough times.
                    force_allow_any_extent = i > MAX_ATTEMPTS_FOR_GRASPING / 2
                    grasp_poses = get_grasp_poses_for_object(
                        self.robot, obj, force_allow_any_extent=force_allow_any_extent
                    )
                    grasp_pose, object_direction = random.choice(grasp_poses)
                    with UndoableContext(self.robot):
                        if hand_collision_fn(grasp_pose):
                            print("Rejecting grasp pose candidate due to collision")
                            continue

                    # Prepare data for the approach later.
                    approach_pos = grasp_pose[0] + object_direction * GRASP_APPROACH_DISTANCE
                    approach_pose = (approach_pos, grasp_pose[1])

                    # If the grasp pose is too far, navigate (and discard this pose - it's aligned for the old pos)
                    yield from self._navigate_if_needed(obj, pos_on_obj=approach_pos)
                    yield from self._navigate_if_needed(obj, pos_on_obj=grasp_pose[0])

                    yield from self._move_hand(grasp_pose)

                    # Since the grasp pose is slightly off the object, we want to move towards the object, around 5cm.
                    # It's okay if we can't go all the way because we run into the object.
                    try:
                        print("Performing grasp approach.")
                        yield from self._move_hand_direct(approach_pose)
                    except MotionPrimitiveError:
                        pass

                    print("Grasping.")
                    yield from self._execute_grasp()

                print("Moving hand back to neutral position.")
                yield from self._reset_hand()

                if self._get_obj_in_hand() == obj:
                    return
            except MotionPrimitiveError as e:
                print("Retrying grasp&reset:", e)

        raise MotionPrimitiveError("Object could not be grasped.")

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

        hand_orientation = self.robot.parts["right_hand"].get_orientation()  # Just keep the current hand orientation.
        desired_hand_pose = (toggle_position, hand_orientation)

        try:
            yield from self._move_hand_direct(desired_hand_pose)
        except MotionPrimitiveError:
            pass  # We can accept some deviation here.

        # Put hand back where it was.
        yield from self._reset_hand()

        if obj.states[object_states.ToggledOn].get_value() != value:
            raise MotionPrimitiveError("Failed to toggle object.")

    def _place_with_predicate(self, obj, predicate):
        obj_in_hand = self._get_obj_in_hand()
        if obj_in_hand is None:
            raise MotionPrimitiveError("Cannot place object if not holding one.")

        for _ in range(MAX_ATTEMPTS_FOR_GRASPING):
            obj_in_hand = self._get_obj_in_hand()
            if obj_in_hand is None:
                raise MotionPrimitiveError("Looks like we might have dropped the object.")

            try:
                obj_pose = self._sample_pose_with_object_and_predicate(predicate, obj_in_hand, obj)
                hand_pose = self._get_hand_pose_for_object_pose(obj_pose)
                yield from self._navigate_if_needed(obj, pos_on_obj=hand_pose[0])
                yield from self._move_hand(hand_pose)
                yield from self._execute_release()
                yield from self._reset_hand()

                if obj_in_hand.states[predicate].get_value(obj):
                    return
            except MotionPrimitiveError as e:
                print("Retrying placement:", e)

        raise MotionPrimitiveError("Object could not be placed.")

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
            raise MotionPrimitiveError("Could not make a hand motion plan.")

        # Follow the plan to navigate.
        print("Plan has %d steps." % len(plan))
        for i, xyz_rpy in enumerate(plan):
            print("Executing grasp plan step %d/%d" % (i + 1, len(plan)))
            pose = (xyz_rpy[:3], p.getQuaternionFromEuler(xyz_rpy[3:]))
            yield from self._move_hand_direct(pose)

    def _move_hand_direct(self, target_pose):
        yield from self._move_hand_direct_relative_to_robot(self._get_pose_in_robot_frame(target_pose))

    def _move_hand_direct_relative_to_robot(self, relative_target_pose):
        for _ in range(MAX_STEPS_FOR_HAND_MOVE):
            action = behavior_ik_controller.get_action(self.robot, hand_target_pose=relative_target_pose)
            if action is None:
                return

            yield action

        # TODO(replayMP): Decide if this is needed.
        raise MotionPrimitiveError("Could not move gripper to desired position.")

    def _execute_grasp(self):
        action = np.zeros(26)
        action[-1] = 1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            # If we're too grasped already, stop.
            if self.robot.parts["right_hand"].trigger_fraction >= 1.0:
                break

            # Otherwise, keep applying the action!
            yield action

        # Do nothing for a bit so that AG can trigger.
        for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
            yield np.zeros(26)

        if self._get_obj_in_hand() is None:
            raise MotionPrimitiveError("Could not grasp object!")

    def _execute_release(self):
        action = np.zeros(26)
        action[-1] = -1.0
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            # If we're too released already, stop.
            trigger_fraction = self.robot.parts["right_hand"].trigger_fraction
            if trigger_fraction <= 0.0:
                break

            # Otherwise, keep applying the action!
            yield action

        # Do nothing for a bit so that AG can trigger.
        for _ in range(MAX_WAIT_FOR_GRASP_OR_RELEASE):
            yield np.zeros(26)

        if self._get_obj_in_hand() is not None or self.robot.parts["right_hand"].trigger_fraction > 0.0:
            raise MotionPrimitiveError("Could not release grasp!")

    def _reset_hand(self):
        # TODO(replayMP): Could we use motion planning here?
        default_pose = p.multiplyTransforms(
            *self.robot.parts["body"].get_position_orientation(), *behavior_robot.RIGHT_HAND_LOC_POSE_TRACKED
        )
        yield from self._move_hand(default_pose)

    def _navigate_to_pose(self, pose_2d):
        # TODO(lowprio-replayMP): Make this less awful.
        start_xy = np.array(self.robot.get_position())[:2]
        end_xy = pose_2d[:2]
        center_xy = (start_xy + end_xy) / 2
        circle_radius = np.linalg.norm(end_xy - start_xy)  # circle with 2x diameter as the distance

        def sample_fn():
            # With some probability, sample from a circle centered around the start & goal.
            within_circle = np.random.rand() > BIRRT_SAMPLING_CIRCLE_PROBABILITY

            while True:
                random_point = np.array(self.scene.get_random_point()[1][:2])

                if within_circle:
                    if np.linalg.norm(random_point - center_xy) < circle_radius:
                        return tuple(random_point)
                else:
                    return tuple(random_point)

        with UndoableContext(self.robot):
            # Note that the plan returned by this planner only contains xy pairs & not yaw.
            plan = plan_base_motion_br(
                robot=self.robot,
                obj_in_hand=self._get_obj_in_hand(),
                end_conf=end_xy,
                sample_fn=sample_fn,
                obstacles=self._get_collision_body_ids(),
            )

        if plan is None:
            raise MotionPrimitiveError("Could not make a navigation plan.")

        # Follow the plan to navigate.
        print("Plan has %d steps." % len(plan))
        for i, xy_pos in enumerate(plan):
            print("Executing navigation plan step %d/%d" % (i + 1, len(plan)))
            low_precision = True if i < len(plan) - 1 else False
            yield from self._navigate_to_pose_direct(xy_pos, low_precision=low_precision)

        # Match the final desired yaw.
        yield from self._rotate_in_place(pose_2d[2])

    def _rotate_in_place(self, yaw, low_precision=False):
        cur_xy_pos = np.array(self.robot.get_position())[:2]

        pose = self._get_robot_pose_from_2d_pose((cur_xy_pos[0], cur_xy_pos[1], yaw))
        for _ in range(MAX_STEPS_FOR_WAYPOINT_NAVIGATION):
            action = behavior_ik_controller.get_action(
                self.robot, body_target_pose=self._get_pose_in_robot_frame(pose), low_precision=low_precision
            )
            if action is None:
                print("Rotate is complete.")
                break

            yield action

    def _navigate_to_pose_direct(self, xy_pos, low_precision=False):
        # First, rotate the robot to face towards the waypoint.
        cur_xy_pos = np.array(self.robot.get_position())[:2]
        displacement = np.array(xy_pos) - cur_xy_pos
        z_rot = np.arctan2(displacement[1], displacement[0])
        yield from self._rotate_in_place(z_rot)

        # Keep the same orientation until the target.
        pose = self._get_robot_pose_from_2d_pose((xy_pos[0], xy_pos[1], z_rot))
        for _ in range(MAX_STEPS_FOR_WAYPOINT_NAVIGATION):
            action = behavior_ik_controller.get_action(
                self.robot, body_target_pose=self._get_pose_in_robot_frame(pose), low_precision=low_precision
            )
            if action is None:
                print("Move is complete.")
                return

            yield action

        # TODO: Do we care if navigation fails?
        # raise MotionPrimitiveError("Could not move robot to desired waypoint.")

    def _navigate_to_obj(self, obj, pos_on_obj=None):
        for _ in range(MAX_ATTEMPTS_FOR_OBJECT_NAVIGATION):
            try:
                yield from self._navigate_to_obj_once(obj, pos_on_obj=pos_on_obj)
                return
            except MotionPrimitiveError as e:
                print("Retrying object navigation: ", e)

        raise MotionPrimitiveError("Could not navigate to object after multiple attempts.")

    def _navigate_to_obj_once(self, obj, pos_on_obj=None):
        if isinstance(obj, RoomFloor):
            # TODO(lowprio-replayMP): Pos-on-obj for the room navigation?
            pose = self._sample_pose_in_room(obj.room_instance)
        else:
            pose = self._sample_pose_near_object(obj, pos_on_obj=pos_on_obj)

        yield from self._navigate_to_pose(pose)

    def _sample_pose_near_object(self, obj, pos_on_obj=None):
        if pos_on_obj is None:
            pos_on_obj = self._sample_position_on_aabb_face(obj)

        pos_on_obj = np.array(pos_on_obj)
        obj_room = self.scene.get_room_instance_by_point(pos_on_obj[:2])
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_NEAR_OBJECT):
            distance = np.random.uniform(0.2, 1.0)
            yaw = np.random.uniform(-np.pi, np.pi)
            pose_2d = np.array(
                [pos_on_obj[0] + distance * np.cos(yaw), pos_on_obj[1] + distance * np.sin(yaw), yaw + np.pi]
            )

            # Check room
            if self.scene.get_room_instance_by_point(pose_2d[:2]) != obj_room:
                print("Candidate position is in the wrong room.")
                continue

            # Check line-of-sight
            # TODO(lowprio-replayMP): Generalize
            pos, _ = self._get_robot_pose_from_2d_pose(pose_2d)
            eye_pos = pos + np.array(behavior_robot.EYE_LOC_POSE_TRACKED[0])
            ray_test_res = p.rayTest(eye_pos, pos_on_obj)

            # TODO(replayMP): Do we need the ray test?
            # if len(ray_test_res) > 0 and ray_test_res[0][0] != obj.get_body_id():
            #     print("Candidate position failed ray test.")
            #     continue

            if not self._test_pose(pose_2d, pos_on_obj=pos_on_obj):
                continue

            return pose_2d

        raise MotionPrimitiveError("Could not find valid position near object.")

    def _sample_position_on_aabb_face(self, target_obj):
        aabb_center, aabb_extent = get_center_extent(target_obj.get_body_id())
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
        # TODO(replayMP): Bias the sampling near the agent.
        for _ in range(MAX_ATTEMPTS_FOR_SAMPLING_POSE_IN_ROOM):
            _, pos = self.scene.get_random_point_by_room_instance(room)
            yaw = np.random.uniform(-np.pi, np.pi)
            pose = (pos[0], pos[1], yaw)
            if self._test_pose(pose):
                return pose

        raise MotionPrimitiveError("Could not find valid position in room.")

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
            )

            if not result:
                raise MotionPrimitiveError("Could not sample position.")

            pos, orn = held_obj.get_position_orientation()
            return pos, orn

    def _navigate_if_needed(self, obj, pos_on_obj=None):
        if pos_on_obj is not None:
            if self._get_dist_from_point_to_shoulder(pos_on_obj) < HAND_DISTANCE_THRESHOLD:
                # No need to navigate.
                return
        elif obj.states[object_states.InReachOfRobot].get_value():
            return

        yield from self._navigate_to_obj(obj, pos_on_obj=pos_on_obj)

    @staticmethod
    def _detect_collision(bodyA, obj_in_hand=None):
        collision = []
        obj_in_hand_id = obj_in_hand.get_body_id() if obj_in_hand is not None else None
        for body_id in range(p.getNumBodies()):
            if body_id == bodyA or body_id == obj_in_hand_id:
                continue
            closest_points = p.getClosestPoints(bodyA, body_id, distance=0.01)
            if len(closest_points) > 0:
                collision.append(body_id)
                break
        return collision

    def _detect_robot_collision(self):
        print("Start collision test.")
        body = self._detect_collision(self.robot.parts["body"].get_body_id())
        if body:
            print("Body has collision with objects ", body)
        left = self._detect_collision(self.robot.parts["left_hand"].get_body_id())
        if left:
            print("Left hand has collision with objects ", left)
        right = self._detect_collision(self.robot.parts["right_hand"].get_body_id(), self._get_obj_in_hand())
        if right:
            print("Right hand has collision with objects ", right)
        print("End collision test.")

        return body or left or right

    def _test_pose(self, pose_2d, pos_on_obj=None):
        with UndoableContext(self.robot):
            self.robot.set_position_orientation(*self._get_robot_pose_from_2d_pose(pose_2d))

            if pos_on_obj is not None:
                hand_distance = self._get_dist_from_point_to_shoulder(pos_on_obj)
                if hand_distance > HAND_DISTANCE_THRESHOLD:
                    print("Candidate position failed shoulder distance test.")
                    return False

            if self._detect_robot_collision():
                print("Candidate position failed collision test.")
                return False

            # TODO(replayMP): Other validations here?
            return True

    def _get_robot_pose_from_2d_pose(self, pose_2d):
        pos = np.array([pose_2d[0], pose_2d[1], BODY_OFFSET_FROM_FLOOR])
        orn = p.getQuaternionFromEuler([0, 0, pose_2d[2]])
        return pos, orn

    def _get_pose_in_robot_frame(self, pose):
        body_pose = self.robot.parts["body"].get_position_orientation()
        world_to_body_frame = p.invertTransform(*body_pose)
        relative_target_pose = p.multiplyTransforms(*world_to_body_frame, *pose)
        return relative_target_pose

    def _get_collision_body_ids(self, include_robot=False):
        ids = []
        for object in self.scene.get_objects():
            if isinstance(object, URDFObject):
                ids.extend(object.body_ids)

        if include_robot:
            ids.append(self.robot.parts["left_hand"].get_body_id())
            ids.append(self.robot.parts["body"].get_body_id())

        return ids

    def _get_dist_from_point_to_shoulder(self, pos):
        shoulder_pos_in_base_frame = np.array(behavior_robot.RIGHT_SHOULDER_REL_POS_UNTRACKED)
        point_in_base_frame = np.array(self._get_pose_in_robot_frame((pos, [0, 0, 0, 1]))[0])
        shoulder_to_hand = point_in_base_frame - shoulder_pos_in_base_frame
        return np.linalg.norm(shoulder_to_hand)

    def _get_hand_pose_for_object_pose(self, desired_pose):
        obj_in_hand = self._get_obj_in_hand()

        assert obj_in_hand is not None

        # Get the object pose & the robot hand pose
        obj_in_world = obj_in_hand.get_position_orientation()
        hand_in_world = self.robot.parts["right_hand"].get_position_orientation()

        # Get the hand pose relative to the obj pose
        world_in_obj = p.invertTransform(*obj_in_world)
        hand_in_obj = p.multiplyTransforms(*world_in_obj, *hand_in_world)

        # Now apply desired obj pose.
        desired_hand_pose = p.multiplyTransforms(*desired_pose, *hand_in_obj)

        return desired_hand_pose
