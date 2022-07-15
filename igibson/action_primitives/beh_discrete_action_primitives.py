import copy
import logging
import random
import time

import gym
import numpy as np
import pybullet as p

from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet
from igibson.action_primitives.beh_discrete_action_primitives_cfg import (
    BEHActionPrimitiveSet,
    ap_object_params,
    aps_per_activity,
)
from igibson.controllers import ControlType, JointController
from igibson.object_states.pose import Pose
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.utils.motion_planning_utils import MotionPlanner
from igibson.utils.transform_utils import mat2euler, quat2mat

logger = logging.getLogger(__name__)


class BEHActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, env, task, scene, robot, arm=None, execute_free_space_motion=False):
        """ """
        super().__init__(env, task, scene, robot)

        if arm is None:
            self.arm = self.robot.default_arm
            logger.info("Using with the default arm: {}".format(self.arm))

        # Checks for the right type of robot and controller to use planned trajectories
        assert robot.grasping_mode == "assisted", "This APs implementation requires assisted grasping to check success"
        if robot.model_name in ["Tiago", "Fetch"]:
            assert not robot.rigid_trunk, "The APs will use the trunk of Fetch/Tiago, it can't be rigid"
        assert isinstance(
            robot._controllers["arm_" + self.arm], JointController
        ), "The arm to use with the primitives must be controlled in joint space"
        assert (
            robot._controllers["arm_" + self.arm].control_type == ControlType.POSITION
        ), "The arm to use with the primitives must be controlled in absolute positions"
        assert not robot._controllers[
            "arm_" + self.arm
        ].use_delta_commands, "The arm to use with the primitives cannot be controlled with deltas"

        self.controller_functions = {
            BEHActionPrimitiveSet.NAVIGATE_TO: self._navigate_to,
            BEHActionPrimitiveSet.PICK: self._pick,
            BEHActionPrimitiveSet.PLACE: self._place,
            BEHActionPrimitiveSet.TOGGLE: self._toggle,
            BEHActionPrimitiveSet.PULL: self._grasp_and_pull,
            BEHActionPrimitiveSet.PUSH: self._push,
        }

        self.action_list = aps_per_activity[self.env.config["task"]]
        self.num_discrete_action = len(self.action_list)
        self.initial_pos_dict = {}
        full_observability_2d_planning = True
        collision_with_pb_2d_planning = True
        self.planner = MotionPlanner(
            self.env,
            optimize_iter=10,
            full_observability_2d_planning=full_observability_2d_planning,
            collision_with_pb_2d_planning=collision_with_pb_2d_planning,
            visualize_2d_planning=False,
            visualize_2d_result=False,
            fine_motion_plan=False,
        )
        self.default_direction = np.array((0.0, 0.0, -1.0))  # default hit normal
        self.execute_free_space_motion = execute_free_space_motion

        # Whether we check if the objects have moved from the previous navigation attempt and do not try to navigate
        # to them if they have (this avoids moving to objects after pick and place)
        self.obj_pose_check = True
        self.task_obj_list = self.env.task.object_scope
        self.skip_base_planning = True
        self.skip_arm_planning = True
        self.is_grasping = False
        self.fast_execution = True

    def get_action_space(self):
        return gym.spaces.Discrete(self.num_discrete_action)

    def apply(self, action_index):
        primitive_obj_pair = self.action_list[action_index]
        return self.controller_functions[primitive_obj_pair[0]](primitive_obj_pair[1])

    def _get_obj_in_hand(self):
        obj_in_hand_id = self.robot._ag_obj_in_hand[self.arm]  # TODO(MP): Expose this interface.
        obj_in_hand = self.scene.objects_by_id[obj_in_hand_id] if obj_in_hand_id is not None else None
        return obj_in_hand

    def _get_still_action(self):
        # The camera and arm(s) and any other absolution joint position controller will be controlled with absolute positions
        state = self.robot.get_joint_states()
        joint_positions = np.array([state[joint_name][0] for joint_name in state])
        action = np.zeros(self.robot.action_dim)
        for controller_name, controller in self.robot._controllers.items():
            if (
                isinstance(controller, JointController)
                and controller.control_type == ControlType.POSITION
                and not controller.use_delta_commands
                and not controller.use_constant_goal_position
            ):
                action_idx = self.robot.controller_action_idx[controller_name]
                joint_idx = self.robot.controller_joint_idx[controller_name]
                logger.debug(
                    "Setting action to absolute position for {} in action dims {} corresponding to joint idx {}".format(
                        controller_name, action_idx, joint_idx
                    )
                )
                action[action_idx] = joint_positions[joint_idx]
        if self.is_grasping:
            # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
            action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0
        return action

    def _execute_ee_path(
        self, path, ignore_failure=False, stop_on_contact=False, reverse_path=False, while_grasping=False
    ):
        for arm_action in path if not reverse_path else reversed(path):
            if stop_on_contact and len(self.robot._find_gripper_contacts(arm=self.arm)[0]) != 0:
                logger.info("Contact detected. Stop motion")
                logger.debug("Contacts {}".format(self.robot._find_gripper_contacts(arm=self.arm)))
                logger.debug("Finger ids {}".format([link.link_id for link in self.robot.finger_links[self.arm]]))
                return
            logger.debug("Executing action {}".format(arm_action))
            full_body_action = self._get_still_action()
            # This assumes the arms are called "arm_"+self.arm. Maybe some robots do not follow this convention
            arm_controller_action_idx = self.robot.controller_action_idx["arm_" + self.arm]
            full_body_action[arm_controller_action_idx] = arm_action

            if while_grasping:
                # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
                full_body_action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0
            yield full_body_action

        if stop_on_contact:
            logger.warning("End of motion and no contact detected")
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.EXECUTION_ERROR, "No contact was made.")
        else:
            logger.debug("End of the path execution")

    def _execute_grasp(self):
        action = self._get_still_action()
        # TODO: Extend to non-binary grasping controllers
        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0

        grasping_steps = 5 if self.fast_execution else 10
        for _ in range(grasping_steps):
            yield action
        grasped_object = self._get_obj_in_hand()
        if grasped_object is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "No object detected in hand after executing grasp.",
            )
        else:
            logger.info("Execution of grasping ended with grasped object {}".format(grasped_object.name))
            self.is_grasping = True

    def _execute_ungrasp(self):
        action = self._get_still_action()

        # TODO: Extend to non-binary grasping controllers
        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        ungrasping_steps = 5 if self.fast_execution else 15
        for idx in range(ungrasping_steps):
            action[self.robot.controller_action_idx["gripper_" + self.arm]] = 0.0
            yield action
        if self._get_obj_in_hand() is not None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "Object still detected in hand after executing release.",
                {"object_in_hand": self._get_obj_in_hand()},
            )
        self.is_grasping = False

    def _navigate_to(self, object_name):
        logger.info("Navigating to object {}".format(object_name))
        params = ap_object_params[BEHActionPrimitiveSet.NAVIGATE_TO][object_name]
        goal_pos = params[:3]

        # If we check whether the object has moved from its initial location. If we check that, and the object has moved
        # more than a threshold, we ignore the command
        moved_distance_threshold = 1e-1
        if self.obj_pose_check:
            if self.env.config["task"] in ["putting_away_Halloween_decorations"]:
                obj_pos = self.env.task.object_scope[object_name].states[Pose].get_value()[0]
                if object_name in ["pumpkin.n.02_1", "pumpkin.n.02_2"]:
                    if object_name not in self.initial_pos_dict:
                        self.initial_pos_dict[object_name] = obj_pos
                    else:
                        moved_distance = np.abs(np.sum(self.initial_pos_dict[object_name] - obj_pos))
                        if moved_distance > moved_distance_threshold:
                            raise ActionPrimitiveError(
                                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                                "Object moved from its initial location",
                                {"object_to_navigate": object_name},
                            )

        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(goal_pos)

        # acquire the base direction
        euler = mat2euler(mat)
        target_yaw = euler[-1] + params[3]

        obj_idx_to_ignore = []
        if self.is_grasping:
            obj_idx_to_ignore = [self.robot._ag_obj_in_hand[self.arm]]

        # TODO: This is for Tiago. The large base doesn't work with the planner we have, the robot collides when it
        # rotates next to the objects
        if self.robot.model_name == "Tiago":
            obj_idx_to_ignore.extend([item.get_body_ids()[0] for item in self.env.scene.objects_by_category["pumpkin"]])
            obj_idx_to_ignore.extend(
                [item.get_body_ids()[0] for item in self.env.scene.objects_by_category["straight_chair"]]
            )

        plan = self.planner.plan_base_motion(
            [obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw],
            plan_full_base_motion=not self.skip_base_planning,
            obj_idx_to_ignore=obj_idx_to_ignore,
        )

        if plan is not None and len(plan) > 0:
            self.planner.visualize_base_path(plan, keep_last_location=True)
        else:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No base path found to object",
                {"object_to_navigate": object_name},
            )
        yield self._get_still_action()
        logger.info("Finished navigating to object: {}".format(object_name))
        return

    def _pick(self, object_name):
        logger.info("Picking object {}".format(object_name))
        # Don't do anything if the object is already grasped.
        object_id = self.task_obj_list[object_name].get_body_ids()[0]  # Assume single body objects
        robot_is_grasping = self.robot.is_grasping(candidate_obj=None)
        robot_is_grasping_obj = self.robot.is_grasping(candidate_obj=object_id)
        if robot_is_grasping == IsGraspingState.TRUE:
            if robot_is_grasping_obj == IsGraspingState.TRUE:
                logger.warning("Robot already grasping the desired object")
                yield np.zeros(self.robot.action_dim)
                return
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                    "Cannot grasp when hand is already full.",
                    {"object": object_name},
                )

        params = ap_object_params[BEHActionPrimitiveSet.PICK][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pre_grasping_distance = 0.0
        plan_full_pre_grasp_motion = not self.skip_arm_planning

        picking_direction = self.default_direction

        if len(params) > 3:
            logger.warning(
                "The number of params indicate that this picking position was made robot-agnostic."
                "Adding finger offset."
            )
            finger_size = self.robot.finger_lengths[self.arm]
            pick_place_pos -= picking_direction * finger_size

        pre_pick_path, interaction_pick_path = self.planner.plan_ee_pick(
            pick_place_pos,
            grasping_direction=picking_direction,
            pre_grasping_distance=pre_grasping_distance,
            plan_full_pre_grasp_motion=plan_full_pre_grasp_motion,
        )

        if (
            pre_pick_path is None
            or len(pre_pick_path) == 0
            or interaction_pick_path is None
            or (len(interaction_pick_path) == 0 and pre_grasping_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pick object",
                {"object_to_pick": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-pick path")
        self.planner.visualize_arm_path(pre_pick_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        if pre_grasping_distance != 0:
            # Then, execute the interaction_pick_path stopping if there is a contact
            logger.info("Executing interaction-pick path")
            yield from self._execute_ee_path(interaction_pick_path, stop_on_contact=True)
        # At the end, close the hand
        logger.info("Executing grasp")
        yield from self._execute_grasp()
        if pre_grasping_distance != 0:
            logger.info("Executing retracting path")
            yield from self._execute_ee_path(
                interaction_pick_path, stop_on_contact=False, reverse_path=True, while_grasping=True
            )
        logger.info("Executing retracting path")
        if plan_full_pre_grasp_motion:
            self.planner.visualize_arm_path(
                pre_pick_path, arm=self.arm, reverse_path=True, grasped_obj_id=object_id, keep_last_location=True
            )
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
                grasped_obj_id=object_id,
            )
        yield self._get_still_action()

        logger.info("Pick action completed")

    def _place(self, object_name):
        logger.info("Placing on object {}".format(object_name))
        params = ap_object_params[BEHActionPrimitiveSet.PLACE][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        np_array = np.array(params[:3])
        np_array[0] += random.uniform(0, 0.2)
        vector = mat @ np_array

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        plan_full_pre_drop_motion = not self.skip_arm_planning

        pre_drop_path, _ = self.planner.plan_ee_drop(
            pick_place_pos, arm=self.arm, plan_full_pre_drop_motion=plan_full_pre_drop_motion
        )

        if pre_drop_path is None or len(pre_drop_path) == 0:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to place object",
                {"object_to_place": object_name},
            )

        # First, teleport the robot to the pre-ungrasp motion
        logger.info("Visualizing pre-place path")
        self.planner.visualize_arm_path(
            pre_drop_path,
            arm=self.arm,
            grasped_obj_id=self.robot._ag_obj_in_hand[self.arm],
            keep_last_location=True,
        )
        yield self._get_still_action()
        # At the end, open the hand
        logger.info("Executing ungrasp")
        yield from self._execute_ungrasp()
        # Then, retract the arm
        logger.info("Executing retracting path")
        if plan_full_pre_drop_motion:  # Visualizing the full path...
            self.planner.visualize_arm_path(
                pre_drop_path,
                arm=self.arm,
                keep_last_location=True,
                reverse_path=True,
            )
        else:  # ... or directly teleporting to the last location
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
        logger.info("Place action completed")

    def _toggle(self, object_name):
        logger.info("Toggling object {}".format(object_name))
        params = ap_object_params[BEHActionPrimitiveSet.TOGGLE][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        toggle_pos = copy.deepcopy(obj_pos)
        toggle_pos[0] += vector[0]
        toggle_pos[1] += vector[1]
        toggle_pos[2] += vector[2]

        pre_toggling_distance = 0.0
        plan_full_pre_toggle_motion = not self.skip_arm_planning

        pre_toggle_path, toggle_interaction_path = self.planner.plan_ee_toggle(
            toggle_pos,
            -np.array(self.default_direction),
            pre_toggling_distance=pre_toggling_distance,
            plan_full_pre_toggle_motion=plan_full_pre_toggle_motion,
        )

        if (
            pre_toggle_path is None
            or len(pre_toggle_path) == 0
            or toggle_interaction_path is None
            or (len(toggle_interaction_path) == 0 and pre_toggling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to toggle object",
                {"object_to_toggle": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-toggle path")
        self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Executing interaction-toggle path")
        yield from self._execute_ee_path(toggle_interaction_path, stop_on_contact=True)

        logger.info("Executing retracting path")
        if plan_full_pre_toggle_motion:
            self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, reverse_path=True, keep_last_location=True)
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
        yield self._get_still_action()

        logger.info("Toggle action completed")

    def _grasp_and_pull(self, object_name):
        logger.info("Pulling object {}".format(object_name))

        robot_is_grasping = self.robot.is_grasping(candidate_obj=None)
        if robot_is_grasping == IsGraspingState.TRUE:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PRE_CONDITION_ERROR,
                "Cannot grasp to pull when hand is already full.",
                {"object": object_name},
            )

        params = ap_object_params[BEHActionPrimitiveSet.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pulling_pos = copy.deepcopy(obj_pos)
        pulling_pos[0] += vector[0]
        pulling_pos[1] += vector[1]
        pulling_pos[2] += vector[2]

        pulling_direction = np.array(params[3:6])
        ee_pulling_orn = p.getQuaternionFromEuler((np.pi / 2, np.pi / 16, 0))
        pre_pulling_distance = 0.1
        pulling_distance = 0.30

        finger_size = self.robot.finger_lengths[self.arm]
        logger.debug("Using finger length {} to adapt action".format(finger_size))
        pulling_pos += pulling_direction * finger_size

        plan_full_pre_pull_motion = not self.skip_arm_planning
        plan_pull_interaction_motion = False

        pre_pull_path, approach_interaction_path, pull_interaction_path = self.planner.plan_ee_pull(
            pulling_location=pulling_pos,
            pulling_direction=pulling_direction,
            ee_pulling_orn=ee_pulling_orn,
            pre_pulling_distance=pre_pulling_distance,
            pulling_distance=pulling_distance,
            plan_full_pre_pull_motion=plan_full_pre_pull_motion,
            pulling_steps=20 if self.fast_execution else 30,
            plan_pull_interaction_motion=plan_pull_interaction_motion,
        )

        if (
            pre_pull_path is None
            or len(pre_pull_path) == 0
            or approach_interaction_path is None
            or (len(approach_interaction_path) == 0 and pre_pulling_distance != 0)
            or plan_pull_interaction_motion
            and pull_interaction_path is None
            or plan_pull_interaction_motion
            and (len(pull_interaction_path) == 0 and pulling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to grasp and pull from object",
                {"object_to_pull": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Visualizing pre-pull path")
        self.planner.visualize_arm_path(pre_pull_path, arm=self.arm)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Executing approaching pull path")
        yield from self._execute_ee_path(approach_interaction_path, stop_on_contact=True)
        # At the end, close the hand
        logger.info("Executing grasp")
        yield from self._execute_grasp()
        # Then, execute the interaction_pull_path
        # Since we may have stopped earlier due to contact, the precomputed path may be wrong
        # We have two options here: 1) (re)plan from the current pose (online planning), or 2) find the closest point
        # in the precomputed trajectory and start the execution there. Implementing option 1)
        logger.info("Replaning interaction pull path and executing")
        current_ee_position = self.robot.get_eef_position(arm=self.arm)
        current_ee_orn = self.robot.get_eef_orientation(arm=self.arm)
        state = self.robot.get_joint_states()
        joint_positions = np.array([state[joint_name][0] for joint_name in state])
        arm_joint_positions = joint_positions[self.robot.controller_joint_idx["arm_" + self.arm]]
        pull_interaction_path = self.planner.plan_ee_straight_line_motion(
            arm_joint_positions,
            current_ee_position,
            pulling_direction,
            ee_orn=current_ee_orn,
            line_length=pulling_distance,
            arm=self.arm,
        )
        if pull_interaction_path is None or len(pull_interaction_path) == 0:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pull object (pulling interaction)",
                {"object_to_pull": object_name},
            )
        yield from self._execute_ee_path(pull_interaction_path, while_grasping=True)
        # Then, open the hand
        logger.info("Executing ungrasp")
        yield from self._execute_ungrasp()
        logger.info("Untuck arm")
        self.planner.visualize_arm_path(
            [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
            arm=self.arm,
            keep_last_location=True,
        )
        yield self._get_still_action()

        logger.info("Pull action completed")

    def _push(self, object_name):
        logger.info("Pushing object {}".format(object_name))
        params = ap_object_params[BEHActionPrimitiveSet.PUSH][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pushing_direction = np.array(params[3:6])

        if len(params) > 6:
            logger.warning(
                "The number of params indicate that this picking position was made robot-agnostic."
                "Adding finger offset."
            )
            finger_size = self.robot.finger_lengths[self.arm]
            pick_place_pos -= pushing_direction * finger_size

        pushing_distance = 0.12
        ee_pushing_orn = np.array(p.getQuaternionFromEuler((0, np.pi / 2, 0)))

        plan_full_pre_push_motion = not self.skip_arm_planning

        pre_push_path, push_interaction_path = self.planner.plan_ee_push(
            pick_place_pos,
            pushing_direction=pushing_direction,
            ee_pushing_orn=ee_pushing_orn,
            pushing_distance=pushing_distance,
            plan_full_pre_push_motion=plan_full_pre_push_motion,
        )

        if (
            pre_push_path is None
            or len(pre_push_path) == 0
            or push_interaction_path is None
            or (len(push_interaction_path) == 0 and pushing_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to push object",
                {"object_to_push": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.info("Pre-push motion")
        self.planner.visualize_arm_path(pre_push_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.info("Pushing interaction")
        yield from self._execute_ee_path(push_interaction_path, stop_on_contact=False)

        logger.info("Tuck arm")
        self.planner.visualize_arm_path(
            [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
            arm=self.arm,
            keep_last_location=True,
        )
        yield self._get_still_action()

        logger.info("Push action completed")
