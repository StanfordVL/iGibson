import copy
import logging
import random
from enum import IntEnum

import gym
import numpy as np
import pybullet as p

from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError, BaseActionPrimitiveSet
from igibson.controllers import ControlType, JointController
from igibson.object_states.pose import Pose
from igibson.robots.manipulation_robot import IsGraspingState
from igibson.utils.motion_planning_utils import MotionPlanner
from igibson.utils.transform_utils import mat2euler, quat2mat

logger = logging.getLogger(__name__)

index_action_mapping = {
    0: "move",
    1: "pick",
    2: "place",
    3: "toggle",
    4: "pull",
    5: "push",
    6: "vis_pick",
    7: "vis_place",
    8: "vis_pull",
    9: "vis_push",
}

skill_object_offset_params = {
    0: {  # skill id: move
        "printer.n.03_1": [-0.7, 0, 0, 0],  # dx, dy, dz, target_yaw
        "table.n.02_1": [0, -0.6, 0, 0.5 * np.pi],
        # Pomaria_1_int, 2
        "hamburger.n.01_1": [0, -0.8, 0, 0.5 * np.pi],
        "hamburger.n.01_2": [0, -0.7, 0, 0.5 * np.pi],
        "hamburger.n.01_3": [0, -0.8, 0, 0.5 * np.pi],
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi],
        "countertop.n.01_1": [0.0, -0.8, 0, 0.5 * np.pi],  # [0.1, 0.5, 0.8 1.0]
        # 'countertop.n.01_1': [[0.0, -0.8, 0, 0.1 * np.pi], [0.0, -0.8, 0, 0.5 * np.pi], [0.0, -0.8, 0, 0.8 * np.pi],],  # [0.1, 0.5, 0.8 1.0]
        # # Ihlen_1_int, 0
        # 'hamburger.n.01_1': [0, 0.8, 0, -0.5 * np.pi],
        # 'hamburger.n.01_2': [0, 0.8, 0, -0.5 * np.pi],
        # 'hamburger.n.01_3': [-0.2, 0.7, 0, -0.6 * np.pi],
        # 'ashcan.n.01_1': [-0.2, -0.5, 0, 0.4 * np.pi],
        # 'countertop.n.01_1': [-0.5, -0.6, 0, 0.5 * np.pi],
        # putting_away_Halloween_decorations
        "pumpkin.n.02_1": [0.4, 0.0, 0.0, 1.0 * np.pi],
        "pumpkin.n.02_2": [0, -0.5, 0, 0.5 * np.pi],
        "cabinet.n.01_1": [0.4, -1.15, 0, 0.5 * np.pi],
    },
    1: {  # pick
        "printer.n.03_1": [-0.2, 0.0, 0.2],  # dx, dy, dz
        # Pomaria_1_int, 2
        "hamburger.n.01_1": [0.0, 0.0, 0.025],
        "hamburger.n.01_2": [
            0.0,
            0.0,
            0.025,
        ],
        "hamburger.n.01_3": [
            0.0,
            0.0,
            0.025,
        ],
        # putting_away_Halloween_decorations
        "pumpkin.n.02_1": [
            0.0,
            0.0,
            0.025,
        ],
        "pumpkin.n.02_2": [
            0.0,
            0.0,
            0.025,
        ],
    },
    2: {  # place
        "table.n.02_1": [0, 0, 0.5],  # dx, dy, dz
        # Pomaria_1_int, 2
        # 'ashcan.n.01_1': [0, 0, 0.5],
        # Ihlen_1_int, 0
        "ashcan.n.01_1": [0, 0, 0.5],
        # putting_away_Halloween_decorations
        # 'cabinet.n.01_1': [0.3, -0.55, 0.25],
        "cabinet.n.01_1": [0.3, -0.60, 0.25],
    },
    3: {  # toggle
        "printer.n.03_1": [-0.3, -0.25, 0.23],  # dx, dy, dz
    },
    4: {  # pull
        "cabinet.n.01_1": [0.35, -0.35, 0.35, -1, 0, 0],  # dx, dy, dz
    },
    5: {  # push
        "cabinet.n.01_1": [0.3, -0.65, 0.35, 1, 0, 0],  # dx, dy, dz
    },
    6: {  # vis_pick
        "hamburger.n.01_1": [0, -0.8, 0, 0.5 * np.pi, 0.0, 0.0, 0.025],
        "hamburger.n.01_2": [
            0,
            -0.7,
            0,
            0.5 * np.pi,
            0.0,
            0.0,
            0.025,
        ],
        "hamburger.n.01_3": [
            0,
            -0.8,
            0,
            0.5 * np.pi,
            0.0,
            0.0,
            0.025,
        ],
        # vis: putting_away_Halloween_decorations
        "pumpkin.n.02_1": [
            0.4,
            0.0,
            0.0,
            1.0 * np.pi,
            0.0,
            0.0,
            0.025,
        ],
        "pumpkin.n.02_2": [
            0,
            -0.5,
            0,
            0.5 * np.pi,
            0.0,
            0.0,
            0.025,
        ],
    },
    7: {  # vis_place
        "ashcan.n.01_1": [0, 0.8, 0, -0.5 * np.pi, 0, 0, 0.5],
        # vis: putting_away_Halloween_decorations
        "cabinet.n.01_1": [0.4, -1.15, 0, 0.5 * np.pi, 0.3, -0.60, 0.25],
    },
    8: {  # vis pull
        "cabinet.n.01_1": [0.3, -0.55, 0.35],  # dx, dy, dz
    },
    9: {  # vis push
        "cabinet.n.01_1": [0.3, -0.8, 0.35],  # dx, dy, dz
    },
}

action_list_installing_a_printer = [
    [0, "printer.n.03_1"],  # skill id, target_obj
    [1, "printer.n.03_1"],
    [0, "table.n.02_1"],
    [2, "table.n.02_1"],
    [3, "printer.n.03_1"],
]

# action_list_throwing_away_leftovers = [
#     [0, 'hamburger.n.01_1'],
#     [1, 'hamburger.n.01_1'],
#     [0, 'ashcan.n.01_1'],
#     [2, 'ashcan.n.01_1'],  # place
#     [0, 'hamburger.n.01_2'],
#     [1, 'hamburger.n.01_2'],
#     [0, 'ashcan.n.01_1'],
#     [2, 'ashcan.n.01_1'],  # place
#     [0, 'hamburger.n.01_3'],
#     [1, 'hamburger.n.01_3'],
#     [0, 'ashcan.n.01_1'],
#     [2, 'ashcan.n.01_1'],  # place
# ]*4

action_list_throwing_away_leftovers_v1 = [
    [0, "hamburger.n.01_1"],
    [1, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [2, "ashcan.n.01_1"],  # place
    [0, "hamburger.n.01_2"],
    [1, "hamburger.n.01_2"],
    [0, "hamburger.n.01_3"],
    [1, "hamburger.n.01_3"],
]

action_list_throwing_away_leftovers_discrete = [
    [0, "countertop.n.01_1", 0],
    [0, "countertop.n.01_1", 1],
    [0, "countertop.n.01_1", 2],
    [6, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [7, "ashcan.n.01_1"],  # place
    [6, "hamburger.n.01_2"],
    [6, "hamburger.n.01_3"],
]

action_list_throwing_away_leftovers = [
    [0, "countertop.n.01_1", 0],
    [6, "hamburger.n.01_1"],
    [0, "ashcan.n.01_1"],
    [7, "ashcan.n.01_1"],  # place
    [6, "hamburger.n.01_2"],
    [6, "hamburger.n.01_3"],
]

# action_list_throwing_away_leftovers = [
#     [0, 'countertop.n.01_1', 0],
#     [6, 'hamburger.n.01_2'],  # 1: 137, 2: 138, 3: 139, plate: 135, ashcan: 140
#     [0, 'ashcan.n.01_1'],
#     [7, 'ashcan.n.01_1'],  # place
#     [0, 'countertop.n.01_1', 1],
#     [6, 'hamburger.n.01_1'],
#     [0, 'ashcan.n.01_1'],
#     [7, 'ashcan.n.01_1'],  # place
#     [0, 'countertop.n.01_1', 2],
#     [6, 'hamburger.n.01_3'],
#     [0, 'ashcan.n.01_1'],
#     [7, 'ashcan.n.01_1'],  # place
# ]

action_list_putting_leftovers_away = [
    [0, "pasta.n.02_1"],
    [1, "pasta.n.02_1"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2"],
    [1, "pasta.n.02_2"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2_3"],
    [1, "pasta.n.02_2_3"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
    [0, "pasta.n.02_2_4"],
    [1, "pasta.n.02_2_4"],
    [0, "countertop.n.01_1"],
    [2, "countertop.n.01_1"],  # place
]

# full set
# action_list_putting_away_Halloween_decorations = [
#     [0, 'cabinet.n.01_1'],  # move
#     [4, 'cabinet.n.01_1'],  # pull
#     [0, 'pumpkin.n.02_1'],  # move
#     [1, 'pumpkin.n.02_1'],  # pick
#     [2, 'cabinet.n.01_1'],  # place
#     [0, 'pumpkin.n.02_2'],  # move
#     [1, 'pumpkin.n.02_2'],  # pick
#     [5, 'cabinet.n.01_1'],  # push
# ]

# full set
action_list_putting_away_Halloween_decorations_v1 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # pick
    # [0, 'cabinet.n.01_1'],  # move
    [2, "cabinet.n.01_1"],  # place
    [0, "pumpkin.n.02_2"],  # move
    [1, "pumpkin.n.02_2"],  # pick
    # [0, 'cabinet.n.01_1'],  # move
    # [2, 'cabinet.n.01_1'],  # place
    [5, "cabinet.n.01_1"],  # push
]

# full sequence
# action_list_putting_away_Halloween_decorations = [
#     # [0, 'cabinet.n.01_1'],  # move
#     # [4, 'cabinet.n.01_1'],  # pull
#     [0, 'pumpkin.n.02_1'],  # move
#     [1, 'pumpkin.n.02_1'],  # pick
#     [0, 'cabinet.n.01_1'],  # move
#     [2, 'cabinet.n.01_1'],  # place
#     [0, 'pumpkin.n.02_2'],  # move
#     [1, 'pumpkin.n.02_2'],  # pick
#     #
#     # [0, 'cabinet.n.01_1'],  # move
#     # [2, 'cabinet.n.01_1'],  # place
#     #
#     # [5, 'cabinet.n.01_1'],  # push
# ] # * 4

# /home/robot/Desktop/behavior/iGibson-dev-jk/igibson/examples/robots/log_dir_his/20220510-001432_putting_away_Halloween_decorations_discrete_rgb_accumReward_m0.01
# wo vis operation
action_list_putting_away_Halloween_decorations_v2 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # pick
    [0, "cabinet.n.01_1"],  # move, repeated
    [2, "cabinet.n.01_1"],  # place
    [0, "pumpkin.n.02_2"],  # move
    [1, "pumpkin.n.02_2"],  # pick
    #
    # [0, 'cabinet.n.01_1'],  # move
    # [2, 'cabinet.n.01_1'],  # place
    #
    [5, "cabinet.n.01_1"],  # push
]  # * 4

# vis version: full sequence
action_list_putting_away_Halloween_decorations_v3 = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # vis pull
    [0, "pumpkin.n.02_1"],  # move
    [6, "pumpkin.n.02_1"],  # vis pick
    [0, "cabinet.n.01_1"],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [0, "pumpkin.n.02_2"],  # move
    [6, "pumpkin.n.02_2"],  # vis pick
    [0, "cabinet.n.01_1"],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [5, "cabinet.n.01_1"],  # vis push
]
# vis version: full set
action_list_putting_away_Halloween_decorations = [
    [0, "cabinet.n.01_1"],  # navigate_to
    [4, "cabinet.n.01_1"],  # pull
    [0, "pumpkin.n.02_1"],  # navigate_to
    [1, "pumpkin.n.02_1"],  # pick
    [2, "cabinet.n.01_1"],  # place
    [0, "pumpkin.n.02_2"],  # navigate_to
    [1, "pumpkin.n.02_2"],  # pick
    [5, "cabinet.n.01_1"],  # push
]

action_list_room_rearrangement = [
    [0, "cabinet.n.01_1"],  # move
    [4, "cabinet.n.01_1"],  # vis pull
    [0, "pumpkin.n.02_1"],  # move
    [1, "pumpkin.n.02_1"],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    [7, "cabinet.n.01_1"],  # vis place
    [0, "pumpkin.n.02_2"],  # move
    [6, "pumpkin.n.02_2"],  # vis pick
    # [0, 'cabinet.n.01_1'],  # move
    # [7, 'cabinet.n.01_1'],  # vis place
    [5, "cabinet.n.01_1"],  # vis push
]

action_dict = {
    "installing_a_printer": action_list_installing_a_printer,
    "throwing_away_leftovers": action_list_throwing_away_leftovers,
    "putting_leftovers_away": action_list_putting_leftovers_away,
    "putting_away_Halloween_decorations": action_list_putting_away_Halloween_decorations,
    "throwing_away_leftovers_discrete": action_list_throwing_away_leftovers_discrete,
    "room_rearrangement": action_list_room_rearrangement,
}


class B1KActionPrimitive(IntEnum):
    NAVIGATE_TO = 0
    PICK = 1
    PLACE = 2
    TOGGLE = 3
    PULL = 4
    PUSH = 5


class B1KActionPrimitives(BaseActionPrimitiveSet):
    def __init__(self, env, task, scene, robot, arm=None, execute_free_space_motion=False):
        super().__init__(env, task, scene, robot)
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
            logger.warning("Using with the default arm: {}".format(self.arm))

        if self.env.config["task"] == "throwing_away_leftovers":
            self.action_list = action_dict["throwing_away_leftovers_discrete"]
            skill_object_offset_params[0]["countertop.n.01_1"] = [
                [0.0, -0.8, 0, 0.1 * np.pi],
                [0.0, -0.8, 0, 0.5 * np.pi],
                [0.0, -0.8, 0, 0.8 * np.pi],
            ]
        else:
            self.action_list = action_dict[self.env.config["task"]]
        self.num_discrete_action = len(self.action_list)
        self.initial_pos_dict = {}  # TODO: what for?
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
        self.obj_pose_check = False
        self.task_obj_list = self.env.task.object_scope
        self.print_log = True
        assert isinstance(
            robot._controllers["arm_" + self.arm], JointController
        ), "The arm to use with the primitives must be controlled in joint space"
        assert (
            robot._controllers["arm_" + self.arm].control_type == ControlType.POSITION
        ), "The arm to use with the primitives must be controlled in absolute positions"
        assert not robot._controllers[
            "arm_" + self.arm
        ].use_delta_commands, "The arm to use with the primitives cannot be controlled with deltas"
        self.skip_base_planning = False
        self.skip_arm_planning = True
        self.is_grasping = False

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
        for controller_name in self.robot._controllers:
            if (
                isinstance(self.robot._controllers[controller_name], JointController)
                and self.robot._controllers[controller_name].control_type == ControlType.POSITION
                and not self.robot._controllers[controller_name].use_delta_commands
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
                logger.warning("Contact detected. Stop motion")
                logger.warning("Contacts {}".format(self.robot._find_gripper_contacts(arm=self.arm)))
                logger.warning("Finger ids {}".format([link.link_id for link in self.robot.finger_links[self.arm]]))
                return
            logger.debug("Executing action {}".format(arm_action))
            full_body_action = np.zeros(self.robot.action_dim)
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
            logger.warning("End of the path execution")

    def _execute_grasp(self):
        action = self._get_still_action()

        # This assumes the grippers are called "gripper_"+self.arm. Maybe some robots do not follow this convention
        action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1.0

        MAX_STEPS_FOR_GRASP_OR_RELEASE = 10
        for _ in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            yield action

        grasped_object = self._get_obj_in_hand()
        if grasped_object is None:
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.EXECUTION_ERROR,
                "No object detected in hand after executing grasp.",
            )
        else:
            logger.warning("Execution of grasping ended with grasped object {}".format(grasped_object.name))
            self.is_grasping = True

    def _execute_ungrasp(self):
        action = self._get_still_action()

        MAX_STEPS_FOR_GRASP_OR_RELEASE = 10
        for idx in range(MAX_STEPS_FOR_GRASP_OR_RELEASE):
            action[self.robot.controller_action_idx["gripper_" + self.arm]] = -1 + float(idx) / float(
                MAX_STEPS_FOR_GRASP_OR_RELEASE - 1
            )
            yield action
        action[self.robot.controller_action_idx["gripper_" + self.arm]] = 0.0
        yield action
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
        logger.warning("Navigating to object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.NAVIGATE_TO][object_name]
        skip_move_flag = False

        # If we check whether the object has moved from its initial location. If we check that, and the object has moved
        # more than a threshold, we ignore the command
        moved_distance_threshold = 1e-1
        if self.obj_pose_check:
            if self.env.config["task"] in ["putting_away_Halloween_decorations"]:
                obj_pos = self.self.env.task.object_scope[object_name].states[Pose].get_value()[0]
                if object_name in ["pumpkin.n.02_1", "pumpkin.n.02_2"]:
                    if object_name not in self.initial_pos_dict:
                        self.initial_pos_dict[object_name] = obj_pos
                    else:
                        moved_distance = np.abs(np.sum(self.initial_pos_dict[object_name] - obj_pos))
                        if np.abs(np.sum(self.initial_pos_dict[object_name] - obj_pos)) > moved_distance_threshold:
                            skip_move_flag = True
                            logger.debug(
                                "Ignoring command to move to object {} because the object has been moved {} m "
                                "from its initial location".format(object_name, moved_distance)
                            )

        if not skip_move_flag:
            obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
            obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

            # process the offset from object frame to world frame
            mat = quat2mat(obj_rot_XYZW)
            vector = mat @ np.array(params[:3])

            # acquire the base direction
            euler = mat2euler(mat)
            target_yaw = euler[-1] + params[3]

            plan = self.planner.plan_base_motion(
                [obj_pos[0] + vector[0], obj_pos[1] + vector[1], target_yaw],
                plan_full_base_motion=not self.skip_base_planning,
            )

            if plan is not None and len(plan) > 0:
                self.planner.visualize_base_path(plan, keep_last_location=True)
            else:
                raise ActionPrimitiveError(
                    ActionPrimitiveError.Reason.PLANNING_ERROR,
                    "No base path found to object",
                    {"object_to_navigate": object_name},
                )

            logger.warning("Moving to object: {}".format(object_name))

        yield self._get_still_action()
        return

    def _pick(self, object_name):
        logger.warning("Picking object {}".format(object_name))
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

        params = skill_object_offset_params[B1KActionPrimitive.PICK][object_name]
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

        pre_pick_path, interaction_pick_path = self.planner.plan_ee_pick(
            pick_place_pos,
            self.default_direction,
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
        logger.warning("Visualizing pre-pick path")
        self.planner.visualize_arm_path(pre_pick_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.warning("Executing interaction-pick path")
        yield from self._execute_ee_path(interaction_pick_path, stop_on_contact=True)
        # At the end, close the hand
        logger.warning("Executing grasp")
        yield from self._execute_grasp()
        logger.warning("Executing retracting path")
        yield from self._execute_ee_path(
            interaction_pick_path, stop_on_contact=False, reverse_path=True, while_grasping=True
        )
        logger.warning("Executing retracting path")
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

        logger.warning("Pick action completed")

    def _place(self, object_name):
        logger.warning("Placing on object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PLACE][object_name]
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

        # First, teleport the robot to the beginning of the pre-pick path
        if plan_full_pre_drop_motion:
            self.planner.visualize_arm_path(
                pre_drop_path,
                arm=self.arm,
                grasped_obj_id=self.robot._ag_obj_in_hand[self.arm],
                keep_last_location=True,
            )
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                grasped_obj_id=self.robot._ag_obj_in_hand[self.arm],
                keep_last_location=True,
            )
        yield self._get_still_action()
        # At the end, close the hand
        yield from self._execute_ungrasp()

        logger.warning("Place action completed")

    def _toggle(self, object_name):
        logger.warning("Toggling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.TOGGLE][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        toggle_pos = copy.deepcopy(obj_pos)
        toggle_pos[0] += vector[0]
        toggle_pos[1] += vector[1]
        toggle_pos[2] += vector[2]

        pre_toggle_distance = 0.0
        plan_full_pre_toggle_motion = not self.skip_arm_planning

        pre_toggle_path, toggle_interaction_path = self.planner.plan_ee_toggle(
            toggle_pos,
            -np.array(self.default_direction),
            pre_toggle_distance=pre_toggle_distance,
            plan_full_pre_toggle_motion=plan_full_pre_toggle_motion,
        )

        if (
            pre_toggle_path is None
            or len(pre_toggle_path) == 0
            or toggle_interaction_path is None
            or (len(toggle_interaction_path) == 0 and pre_toggle_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to toggle object",
                {"object_to_toggle": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.warning("Visualizing pre-toggle path")
        self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.warning("Executing interaction-toggle path")
        yield from self._execute_ee_path(toggle_interaction_path, stop_on_contact=True)

        logger.warning("Executing retracting path")
        if plan_full_pre_toggle_motion:
            self.planner.visualize_arm_path(pre_toggle_path, arm=self.arm, reverse_path=True, keep_last_location=True)
        else:
            self.planner.visualize_arm_path(
                [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
                arm=self.arm,
                keep_last_location=True,
            )
        yield self._get_still_action()

        logger.warning("Toggle action completed")

    def _pull(self, object_name):
        logger.warning("Pulling object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PULL][object_name]
        obj_pos = self.task_obj_list[object_name].states[Pose].get_value()[0]
        obj_rot_XYZW = self.task_obj_list[object_name].states[Pose].get_value()[1]

        # process the offset from object frame to world frame
        mat = quat2mat(obj_rot_XYZW)
        vector = mat @ np.array(params[:3])

        pick_place_pos = copy.deepcopy(obj_pos)
        pick_place_pos[0] += vector[0]
        pick_place_pos[1] += vector[1]
        pick_place_pos[2] += vector[2]

        pulling_direction = np.array(params[3:6])
        ee_pulling_orn = p.getQuaternionFromEuler((np.pi / 2, 0, 0))
        pre_pulling_distance = 0.10
        pulling_distance = 0.30

        plan_full_pre_pull_motion = not self.skip_arm_planning

        pre_pull_path, approach_interaction_path, pull_interaction_path = self.planner.plan_ee_pull(
            pulling_location=pick_place_pos,
            pulling_direction=pulling_direction,
            ee_pulling_orn=ee_pulling_orn,
            pre_pulling_distance=pre_pulling_distance,
            pulling_distance=pulling_distance,
            plan_full_pre_pull_motion=plan_full_pre_pull_motion,
            pulling_steps=15,
        )

        if (
            pre_pull_path is None
            or len(pre_pull_path) == 0
            or approach_interaction_path is None
            or (len(approach_interaction_path) == 0 and pre_pulling_distance != 0)
            or pull_interaction_path is None
            or (len(pull_interaction_path) == 0 and pulling_distance != 0)
        ):
            raise ActionPrimitiveError(
                ActionPrimitiveError.Reason.PLANNING_ERROR,
                "No arm path found to pull object",
                {"object_to_pull": object_name},
            )

        # First, teleport the robot to the beginning of the pre-pick path
        logger.warning("Visualizing pre-pull path")
        self.planner.visualize_arm_path(pre_pull_path, arm=self.arm)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.warning("Executing approaching pull path")
        yield from self._execute_ee_path(approach_interaction_path, stop_on_contact=True)
        # At the end, close the hand
        logger.warning("Executing grasp")
        yield from self._execute_grasp()
        # Then, execute the interaction_pull_path
        # Since we may have stopped earlier due to contact, the precomputed path may be wrong
        # We have two options here: 1) (re)plan from the current pose (online planning), or 2) find the closest point
        # in the precomputed trajectory and start the execution there. Implementing option 1)
        logger.warning("Replaning interaction pull path and executing")
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
        yield from self._execute_ee_path(pull_interaction_path, while_grasping=True)
        # Then, open the hand
        logger.warning("Executing ungrasp")
        yield from self._execute_ungrasp()
        logger.warning("Tuck arm")
        self.planner.visualize_arm_path(
            [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
            arm=self.arm,
            keep_last_location=True,
        )
        yield self._get_still_action()

        logger.warning("Pull action completed")

    def _push(self, object_name):
        logger.warning("Pushing object {}".format(object_name))
        params = skill_object_offset_params[B1KActionPrimitive.PUSH][object_name]
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

        pushing_distance = 0.3
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
        logger.warning("Pre-push motion")
        self.planner.visualize_arm_path(pre_push_path, arm=self.arm, keep_last_location=True)
        yield self._get_still_action()
        # Then, execute the interaction_pick_path stopping if there is a contact
        logger.warning("Pushing interaction")
        yield from self._execute_ee_path(push_interaction_path, stop_on_contact=False)

        logger.warning("Tuck arm")
        self.planner.visualize_arm_path(
            [self.robot.untucked_default_joint_pos[self.robot.controller_joint_idx["arm_" + self.arm]]],
            arm=self.arm,
            keep_last_location=True,
        )
        yield self._get_still_action()

        logger.warning("Push action completed")
