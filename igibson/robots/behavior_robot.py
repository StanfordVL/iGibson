"""
BehaviorRobot class that can be used in VR as an avatar, or as a robot.
It has two hands, a body and a head link, so is very close to a humanoid avatar.

Takes in a numpy action space each frame to update its positions.

Action space (all non-normalized values that will be clipped if they are too large)
* See init function for various clipping thresholds for velocity, angular velocity and local position
Body:
- 6DOF pose delta - relative to body frame from previous frame
Eye:
- 6DOF pose delta - relative to body frame (where the body will be after applying this frame's action)
Left hand, right hand (in that order):
- 6DOF pose delta - relative to body frame (same as above)
- Trigger fraction delta
- Action reset value

Total size: 28
"""
import itertools
import logging
import os
from abc import ABCMeta, abstractmethod

import numpy as np
import pybullet as p
from future.utils import with_metaclass

from igibson import assets_path
from igibson.object_states.utils import clear_cached_states
from igibson.objects.visual_marker import VisualMarker
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.locomotion_robot import LocomotionRobot
from igibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from igibson.robots.robot_base import Virtual6DOFJoint, VirtualJoint
from igibson.utils.constants import SPECIAL_COLLISION_GROUPS, SimulatorMode, get_collision_group_mask

log = logging.getLogger(__name__)

# Part offset parameters
DEFAULT_BODY_OFFSET_FROM_FLOOR = 0.55

EYE_LOC_POSE_UNTRACKED = ([0.05, 0, 0], [0, 0, 0, 1])
RIGHT_HAND_LOC_POSE_UNTRACKED = ([0.1, -0.12, -0.4], [-0.7, 0.7, 0.0, 0.15])
LEFT_HAND_LOC_POSE_UNTRACKED = ([0.1, 0.12, -0.4], [0.7, 0.7, 0.0, 0.15])

EYE_LOC_POSE_TRACKED = ([0.05, 0, 0.4], [0, 0, 0, 1])
RIGHT_HAND_LOC_POSE_TRACKED = ([0.1, -0.12, 0.05], [-0.7, 0.7, 0.0, 0.15])
LEFT_HAND_LOC_POSE_TRACKED = ([0.1, 0.12, 0.05], [0.7, 0.7, 0.0, 0.15])

# Body parameters
BODY_HEIGHT_RANGE = (0, 2)  # meters. The body is allowed to clip the floor by about a half.
BODY_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
BODY_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
BODY_MASS = 15  # body mass in kg
BODY_MOVING_FORCE = BODY_MASS * 500
BODY_NO_COLLISION_CATEGORIES = frozenset(["floors", "carpet"])

# Hand parameters
HAND_BASE_ROTS = {"right": p.getQuaternionFromEuler([0, 160, -80]), "left": p.getQuaternionFromEuler([0, 160, 80])}
HAND_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
HAND_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
HAND_DISTANCE_THRESHOLD = 1.2  # distance threshold in meters
HAND_GHOST_HAND_APPEAR_THRESHOLD = 0.15
HAND_FRICTION = 2.5
THUMB_2_POS = np.array([0, -0.02, -0.05])
THUMB_1_POS = np.array([0, -0.015, -0.02])
PALM_CENTER_POS = np.array([0, -0.04, 0.01])
PALM_BASE_POS = np.array([0, 0, 0.015])
FINGER_TIP_POS = np.array([0, -0.025, -0.055])
HAND_LIFTING_FORCE = 300

# Hand link index constants
FINGER_MID_LINK_NAMES = ("Tproximal", "Iproximal", "Mproximal", "Rproximal", "Pproximal")
FINGER_TIP_LINK_NAMES = ("Tmiddle", "Imiddle", "Mmiddle", "Rmiddle", "Pmiddle")
THUMB_LINK_NAME = "Tmiddle"

# Head parameters
HEAD_LINEAR_VELOCITY = 0.3  # linear velocity thresholds in meters/frame
HEAD_ANGULAR_VELOCITY = 1  # angular velocity thresholds in radians/frame
HEAD_DISTANCE_THRESHOLD = 0.5  # distance threshold in meters

# RL mode action scaling
LOWER_LIMITS_POSITION_COEFFICIENT = 0.5
LOWER_LIMITS_VELOCITY_COEFFICIENT = 0.01


class BehaviorRobot(ManipulationRobot, LocomotionRobot, ActiveCameraRobot):
    """
    A robot that is designed for use with a Virtual Reality kit for collecting human demonstrations. It consists of
    four parts: an elliptical body, two floating hands, and a floating and collisionless head.
    """

    DEFAULT_RENDERING_PARAMS = {
        "use_pbr": False,
        "use_pbr_mapping": False,
        "shadow_caster": True,
    }

    def __init__(
        self,
        hands=("left", "right"),
        use_ghost_hands=True,
        normal_color=True,
        show_visual_head=True,
        use_tracked_body=True,
        reset_joint_pos=None,
        base_name="body",
        grasping_mode="assisted",
        higher_limits=False,
        legacy_proprioception=False,
        **kwargs,
    ):
        """
        :param hands: list containing left, right or no hands
        :param use_ghost_hands: Whether ghost hands (e.g. red, collisionless hands indicating where the user is trying
            to move the hand when the current and requested positions differ by more than a threshold) should be shown.
        :param normal_color: whether to use normal color (grey) (when True) or alternative color (blue-tinted). This is
            useful for when there are multiple robots in the scene.
        :param show_visual_head: whether to render a head for the BehaviorRobot. The head does not have collisions.
        :param use_tracked_body: whether the robot is intended for use with a VR kit that has a body tracker or not.
            Different robot models are loaded based on this value. True is recommended.
        :param reset_joint_pos: None or str or Array[float], if specified, should be the joint positions that the robot
            should be set to during a reset. If str, should be one of {tuck, untuck}, corresponds to default
            configurations for un/tucked modes. If None (default), self.default_joint_pos (untuck mode) will be used
            instead.
        :param base_name: None or str, robot link name that will represent the entire robot's frame of reference. If not None,
            this should correspond to one of the link names found in this robot's corresponding URDF / MJCF file.
            None defaults to the base link name used in @model_file
        :param grasping_mode: None or str, One of {"physical", "assisted", "sticky"}.
            If "physical", no assistive grasping will be applied (relies on contact friction + finger force).
            If "assisted", will magnetize any object touching and within the gripper's fingers.
            If "sticky", will magnetize any object touching the gripper's fingers.
        :param higher_limits: bool, indicating whether the limbs' velocity and position limits should be set
            in the extended mode which allows for the instant responsiveness required for VR. Defaults to True.
            learning-based agents should set this parameter to False for a more realistic capability set.
        :param legacy_proprioception: Use the proprioception dimensions and dict keys that the BehaviorRobot had at
            the BEHAVIOR challenge, rather than the ones automatically generated by underlying robot classes.
        :param **kwargs: see ManipulationRobot, LocomotionRobot, ActiveCameraRobot
        """
        assert reset_joint_pos is None, "BehaviorRobot doesn't support hand-specified reset_joint_pos"
        assert base_name == "body", "BehaviorRobot needs the base_name to be 'body'"

        # Basic parameters
        self.hands = hands
        self.use_tracked_body = use_tracked_body
        self.use_ghost_hands = use_ghost_hands
        self.normal_color = normal_color
        self.show_visual_head = show_visual_head
        self._legacy_proprioception = legacy_proprioception

        self._position_limit_coefficient = 1 if higher_limits else LOWER_LIMITS_POSITION_COEFFICIENT
        self._velocity_limit_coefficient = 1 if higher_limits else LOWER_LIMITS_VELOCITY_COEFFICIENT

        super(BehaviorRobot, self).__init__(
            reset_joint_pos=reset_joint_pos, base_name=base_name, grasping_mode=grasping_mode, **kwargs
        )

        # Set up body parts
        self._parts = dict()

        if "left" in self.hands:
            self._parts["left_hand"] = BRHand(self, hand="left")
        if "right" in self.hands:
            self._parts["right_hand"] = BRHand(self, hand="right")

        self._parts["body"] = BRBody(self)
        self._parts["eye"] = BREye(self)

    @property
    def model_name(self):
        return "BehaviorRobot"

    @property
    def model_file(self):
        raise ValueError("BehaviorRobot does not have a single model. It should be loaded using its custom loader.")

    def _load(self, simulator):
        """
        Loads this BehaviorRobot into the simulation. Should return a list of unique body IDs corresponding
        to this model.

        :param simulator: Simulator, iGibson simulator reference

        :return Array[int]: List of unique pybullet IDs corresponding to this robot.
        """
        # A persistent reference to simulator is needed for AG in ManipulationRobot
        self.simulator = simulator

        # Set the control frequency if one was not provided.
        expected_control_freq = 1.0 / simulator.render_timestep
        if self.control_freq is None:
            log.debug(
                "Control frequency is None - being set to default of 1 / render_timestep: %.4f", expected_control_freq
            )
            self.control_freq = expected_control_freq
        else:
            assert np.isclose(
                expected_control_freq, self.control_freq
            ), "Stored control frequency does not match environment's render timestep."

        # TODO: Remove hacky fix - constructor/config should contain this data.
        if self.simulator.mode == SimulatorMode.VR:
            assert (
                self.use_tracked_body == self.simulator.vr_settings.using_tracked_body
            ), "Robot and VR config do not match in terms of whether to use tracked body. Please update either config."

        return [id for part in self._parts.values() for id in part.load(simulator)]

    def load(self, simulator):
        """Do the regular load, then update the collision filters."""
        body_ids = super(BehaviorRobot, self).load(simulator)
        self._parts["body"].set_body_collision_filters()
        return body_ids

    def _setup_virtual_joints(self):
        virtual_joints = []

        # This joint is a bit weird - it's between the body and the body. We can't do world-body since that would make
        # the actions in the world frame rather than the base frame. Instead we do this trick. The absolute value set
        # for the body-body joint is interpreted by the body as the displacement it should make, in its own frame. As
        # a result, a zero value for this joint means staying in place, and a non-zero action is a delta pose in the
        # body frame.
        virtual_joints.extend(
            Virtual6DOFJoint(
                joint_name="body__body",
                parent_link=self.base_link,
                child_link=self.base_link,
                command_callback=self._parts["body"].command_position,
                reset_callback=self._parts["body"].reset_position,
                lower_limits=[None, None, None, None, None, None],
                upper_limits=[None, None, None, None, None, None],
            ).get_joints()
        )

        virtual_joints.extend(
            Virtual6DOFJoint(
                joint_name="neck__camera",
                parent_link=self.links["neck"],
                child_link=self.links["eyes"],
                command_callback=self._parts["eye"].command_position,
                reset_callback=self._parts["eye"].command_position,  # reset and command are the same for eye
                lower_limits=[-HEAD_DISTANCE_THRESHOLD * self._position_limit_coefficient] * 3 + [None] * 3,
                upper_limits=[HEAD_DISTANCE_THRESHOLD * self._position_limit_coefficient] * 3 + [None] * 3,
            ).get_joints(),
        )

        for arm_name, arm_link in self.eef_links.items():
            virtual_joints.extend(
                Virtual6DOFJoint(
                    joint_name="%s_shoulder__%s" % (arm_name, arm_name),
                    parent_link=self.links["%s_shoulder" % arm_name],
                    child_link=arm_link,
                    command_callback=self._parts[arm_name].command_position,
                    reset_callback=self._parts[arm_name].reset_position,
                    lower_limits=[-HAND_DISTANCE_THRESHOLD * self._position_limit_coefficient] * 3 + [None] * 3,
                    upper_limits=[HAND_DISTANCE_THRESHOLD * self._position_limit_coefficient] * 3 + [None] * 3,
                ).get_joints(),
            )
            virtual_joints.append(
                VirtualJoint(
                    joint_name="reset_%s" % arm_name,
                    joint_type=p.JOINT_PRISMATIC,
                    get_state_callback=lambda: (0, 0, 0),
                    set_pos_callback=self._parts[arm_name].command_reset,
                    reset_pos_callback=lambda _: None,  # don't need to reset joint for the reset button
                    lower_limit=0,
                    upper_limit=1,
                )
            )

        return virtual_joints

    def set_position_orientation(self, pos, orn):
        self._parts["body"].set_position_orientation(pos, orn)
        self._parts["body"].new_pos, self._parts["body"].new_orn = pos, orn

        # Local transforms for hands and eye
        if self.use_tracked_body:
            left_hand_loc_pose = LEFT_HAND_LOC_POSE_TRACKED
            right_hand_loc_pose = RIGHT_HAND_LOC_POSE_TRACKED
            eye_loc_pose = EYE_LOC_POSE_TRACKED
        else:
            left_hand_loc_pose = LEFT_HAND_LOC_POSE_UNTRACKED
            right_hand_loc_pose = RIGHT_HAND_LOC_POSE_UNTRACKED
            eye_loc_pose = EYE_LOC_POSE_UNTRACKED

        left_hand_pos, left_hand_orn = p.multiplyTransforms(pos, orn, left_hand_loc_pose[0], left_hand_loc_pose[1])
        self._parts["left_hand"].set_position_orientation(left_hand_pos, left_hand_orn)
        right_hand_pos, right_hand_orn = p.multiplyTransforms(pos, orn, right_hand_loc_pose[0], right_hand_loc_pose[1])
        self._parts["right_hand"].set_position_orientation(right_hand_pos, right_hand_orn)
        eye_pos, eye_orn = p.multiplyTransforms(pos, orn, eye_loc_pose[0], eye_loc_pose[1])
        self._parts["eye"].set_position_orientation(eye_pos, eye_orn)

        clear_cached_states(self)

    @property
    def default_proprio_obs(self):
        if self._legacy_proprioception:
            return [
                "left_hand_position_local",
                "left_hand_orientation_local",
                "right_hand_position_local",
                "right_hand_orientation_local",
                "eye_position_local",
                "eye_orientation_local",
                "left_hand_trigger_fraction",
                "left_hand_is_grasping",
                "right_hand_trigger_fraction",
                "right_hand_is_grasping",
            ]

        # Otherwise just return the default observations
        return super().default_proprio_obs

    def _get_proprioception_dict(self):
        if self._legacy_proprioception:
            # In the legacy proprioception mode, we use some custom labels and fields.
            state = {}

            # Get all the part poses first.
            for part_name, part in self._parts.items():
                if part_name == "body":
                    continue
                part_pos, part_orn = part.get_local_position_orientation()
                state[f"{part_name}_position_local"] = part_pos
                state[f"{part_name}_orientation_local"] = p.getEulerFromQuaternion(part_orn)

            # Get grasping information.
            for arm in self.arm_names:
                finger_pos = self.joint_positions[self.gripper_control_idx[arm]]
                min_pos = self.joint_lower_limits[self.gripper_control_idx[arm]]
                max_pos = self.joint_upper_limits[self.gripper_control_idx[arm]]
                trigger_fraction = np.max((finger_pos - min_pos) / (max_pos - min_pos))
                state[f"{arm}_trigger_fraction"] = [np.clip(trigger_fraction, 0, 1)]
                state[f"{arm}_is_grasping"] = [self.is_grasping(arm=arm)]

            return state

        return super()._get_proprioception_dict()

    def set_poses(self, poses):
        assert len(poses) == len(self._parts), "Number of poses (%d) does not match number of parts (%d)" % (
            len(poses),
            len(self._parts),
        )
        bid_to_pose = {bid: pose for bid, pose in zip(self.get_body_ids(), poses)}
        part_names = ["body"] + list(set(self._parts.keys()) - {"body"})  # Make sure we do body first
        for part_name in part_names:
            part_pose = bid_to_pose[self._parts[part_name].body_id]
            self._parts[part_name].set_position_orientation(*part_pose)

        clear_cached_states(self)

    def reset(self):
        # Move the constraint for each part to the default position.
        self.set_position_orientation(*self.get_position_orientation())

    def keep_still(self):
        # Move the constraint for each part to its current position.
        for part in self._parts.values():
            part.set_position_orientation(*part.get_position_orientation())

        super(BehaviorRobot, self).keep_still()

    def dump_config(self):
        """Dump robot config"""
        parent_info = super(BehaviorRobot, self).dump_config()
        this_info = {
            "hands": self.hands,
            "use_ghost_hands": self.use_ghost_hands,
            "normal_color": self.normal_color,
            "show_visual_head": self.show_visual_head,
            "use_tracked_body": self.use_tracked_body,
        }
        assert set(parent_info.keys()).isdisjoint(this_info.keys()), "Overlapping keys found in config."
        this_info.update(parent_info)

        return this_info

    def dump_state(self):
        dump = super(BehaviorRobot, self).dump_state()

        for part_name, part in self._parts.items():
            dump["part_" + part_name] = part.dump_state()

        return dump

    def load_state(self, dump):
        super(BehaviorRobot, self).load_state(dump)

        for part_name, part in self._parts.items():
            part.load_state(dump["part_" + part_name])

    def get_body_ids(self):
        return [part.body_id for part in self._parts.values()]

    @property
    def n_arms(self):
        return len(self.hands)

    @property
    def arm_names(self):
        return [hand + "_hand" for hand in self.hands]

    @property
    def eef_link_names(self):
        return {arm_name: arm_name + "_base" for arm_name in self.arm_names}

    @property
    def finger_link_names(self):
        return {
            arm: [
                "%s_%s" % (arm, link_name)
                for link_name in itertools.chain(FINGER_MID_LINK_NAMES, FINGER_TIP_LINK_NAMES)
            ]
            for arm in self.arm_names
        }

    @property
    def finger_joint_names(self):
        return {
            arm: (
                # Get the palm-to-proximal joints.
                ["%s_%s__%s_palm" % (arm, to_link, arm) for to_link in FINGER_MID_LINK_NAMES]
                +
                # Get the proximal-to-tip joints.
                [
                    "%s_%s__%s_%s" % (arm, to_link, arm, from_link)
                    for from_link, to_link in zip(FINGER_MID_LINK_NAMES, FINGER_TIP_LINK_NAMES)
                ]
            )
            for arm in self.arm_names
        }

    @property
    def arm_control_idx(self):
        joints = list(self.joints.keys())
        return {
            arm: np.array(
                [
                    joints.index("%s_shoulder__%s_%s" % (arm, arm, component))
                    for component in Virtual6DOFJoint.COMPONENT_SUFFIXES
                ]
            )
            for arm in self.arm_names
        }

    @property
    def base_control_idx(self):
        joints = list(self.joints.keys())
        return np.array(
            [joints.index("body__body_%s" % (component)) for component in Virtual6DOFJoint.COMPONENT_SUFFIXES]
        )

    @property
    def camera_control_idx(self):
        joints = list(self.joints.keys())
        return np.array(
            [joints.index("neck__camera_%s" % (component)) for component in Virtual6DOFJoint.COMPONENT_SUFFIXES]
        )

    @property
    def reset_control_idx(self):
        joints = list(self.joints.keys())
        return {arm: np.array([joints.index("reset_%s" % arm)]) for arm in self.arm_names}

    @property
    def gripper_control_idx(self):
        joints = list(self.joints.values())
        return {
            arm: np.array([joints.index(joint) for joint in finger_joints])
            for arm, finger_joints in self.finger_joints.items()
        }

    @property
    def control_limits(self):
        return {"position": (self.joint_lower_limits, self.joint_upper_limits), "has_limit": self.joint_has_limits}

    @property
    def controller_order(self):
        # Assumes we have arm(s) and corresponding gripper(s)
        controllers = ["base", "camera"]
        for arm in self.arm_names:
            controllers += ["arm_{}".format(arm), "gripper_{}".format(arm), "reset_{}".format(arm)]

        return controllers

    @property
    def _default_controllers(self):
        controllers = {
            "base": "JointController",
            "camera": "JointController",
        }
        controllers.update({"arm_%s" % arm: "JointController" for arm in self.arm_names})
        controllers.update({"gripper_%s" % arm: "MultiFingerGripperController" for arm in self.arm_names})
        controllers.update({"reset_%s" % arm: "JointController" for arm in self.arm_names})
        return controllers

    @property
    def _default_arm_controller_configs(self):
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self.control_freq,
                "control_limits": self.control_limits,
                "use_delta_commands": True,
                "motor_type": "position",
                "compute_delta_in_quat_space": [(3, 4, 5)],
                "joint_idx": self.arm_control_idx[arm],
                "command_input_limits": (
                    np.array([-HAND_LINEAR_VELOCITY] * 3 + [-HAND_ANGULAR_VELOCITY] * 3)
                    * self._velocity_limit_coefficient,
                    np.array([-HAND_LINEAR_VELOCITY] * 3 + [-HAND_ANGULAR_VELOCITY] * 3)
                    * self._velocity_limit_coefficient,
                ),
                "command_output_limits": None,
            }
        return dic

    @property
    def _default_gripper_joint_controller_configs(self):
        # The use case for the joint controller for the BehaviorRobot is supporting the VR action space. We configure
        # this accordingly.
        dic = super(BehaviorRobot, self)._default_gripper_joint_controller_configs

        for arm in self.arm_names:
            # Compute the command output limits that would allow -1 to fully open and 1 to fully close.
            joint_lower_limits = self.control_limits["position"][0][self.gripper_control_idx[arm]]
            joint_upper_limits = self.control_limits["position"][1][self.gripper_control_idx[arm]]
            ranges = joint_upper_limits - joint_lower_limits
            output_ranges = (-ranges, ranges)
            dic[arm].update(
                {
                    "motor_type": "position",
                    "parallel_mode": True,
                    "inverted": True,
                    "command_input_limits": (0, 1),
                    "command_output_limits": output_ranges,
                    "use_delta_commands": True,
                }
            )
        return dic

    @property
    def _default_base_controller_configs(self):
        dic = {
            "name": "JointController",
            "control_freq": self.control_freq,
            "control_limits": self.control_limits,
            "use_delta_commands": False,
            "motor_type": "position",
            "compute_delta_in_quat_space": [(3, 4, 5)],
            "joint_idx": self.base_control_idx,
            "command_input_limits": (
                np.array([-BODY_LINEAR_VELOCITY] * 3 + [-BODY_ANGULAR_VELOCITY] * 3) * self._velocity_limit_coefficient,
                np.array([BODY_LINEAR_VELOCITY] * 3 + [BODY_ANGULAR_VELOCITY] * 3) * self._velocity_limit_coefficient,
            ),
            "command_output_limits": None,
        }
        return dic

    @property
    def _default_camera_controller_configs(self):
        dic = {
            "name": "JointController",
            "control_freq": self.control_freq,
            "control_limits": self.control_limits,
            "use_delta_commands": True,
            "motor_type": "position",
            "compute_delta_in_quat_space": [(3, 4, 5)],
            "joint_idx": self.camera_control_idx,
            "command_input_limits": (
                np.array([-HEAD_LINEAR_VELOCITY] * 3 + [-HEAD_ANGULAR_VELOCITY] * 3) * self._velocity_limit_coefficient,
                np.array([HEAD_LINEAR_VELOCITY] * 3 + [HEAD_ANGULAR_VELOCITY] * 3) * self._velocity_limit_coefficient,
            ),
            "command_output_limits": None,
        }
        return dic

    @property
    def _default_reset_controller_configs(self):
        dic = {}
        for arm in self.arm_names:
            dic[arm] = {
                "name": "JointController",
                "control_freq": self.control_freq,
                "control_limits": self.control_limits,
                "motor_type": "position",
                "use_delta_commands": False,
                "joint_idx": self.reset_control_idx[arm],
                "command_input_limits": None,
            }
        return dic

    @property
    def _default_controller_config(self):
        controllers = {
            "base": {"JointController": self._default_base_controller_configs},
            "camera": {"JointController": self._default_camera_controller_configs},
        }
        controllers.update(
            {"arm_%s" % arm: {"JointController": self._default_arm_controller_configs[arm]} for arm in self.arm_names}
        )
        controllers.update(
            {
                "gripper_%s"
                % arm: {
                    "NullGripperController": self._default_gripper_null_controller_configs[arm],
                    "MultiFingerGripperController": self._default_gripper_multi_finger_controller_configs[arm],
                    "JointController": self._default_gripper_joint_controller_configs[arm],
                }
                for arm in self.arm_names
            }
        )
        controllers.update(
            {
                "reset_%s" % arm: {"JointController": self._default_reset_controller_configs[arm]}
                for arm in self.arm_names
            }
        )
        return controllers

    @property
    def default_joint_pos(self):
        # NOT actually being used for reset, because the reset function is overwritten
        return None

    def _create_discrete_action_space(self):
        raise ValueError("BehaviorRobot does not support discrete actions!")

    @property
    def assisted_grasp_start_points(self):
        side_coefficients = {"left_hand": np.array([1, -1, 1]), "right_hand": np.array([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name="%s_palm" % arm, position=PALM_BASE_POS),
                GraspingPoint(link_name="%s_palm" % arm, position=PALM_CENTER_POS * side_coefficients[arm]),
                GraspingPoint(
                    link_name="%s_%s" % (arm, THUMB_LINK_NAME), position=THUMB_1_POS * side_coefficients[arm]
                ),
                GraspingPoint(
                    link_name="%s_%s" % (arm, THUMB_LINK_NAME), position=THUMB_2_POS * side_coefficients[arm]
                ),
            ]
            for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        side_coefficients = {"left_hand": np.array([1, -1, 1]), "right_hand": np.array([1, 1, 1])}
        return {
            arm: [
                GraspingPoint(link_name="%s_%s" % (arm, finger), position=FINGER_TIP_POS * side_coefficients[arm])
                for finger in FINGER_TIP_LINK_NAMES
            ]
            for arm in self.arm_names
        }

    @property
    def gripper_link_to_grasp_point(self):
        return {arm: PALM_CENTER_POS * (1 if arm == "right_hand" else -1) for arm in self.arm_names}


class BRPart(with_metaclass(ABCMeta, object)):
    """This is the interface that all BehaviorRobot parts must implement."""

    def __init__(self):
        """
        Create an object instance with the minimum information of class ID and rendering parameters.
        """
        self.renderer_instances = []

        self.new_pos = None
        self.new_orn = None

        self._loaded = False
        self.body_id = None

    def load(self, simulator):
        """Load object into pybullet and return list of loaded body ids."""
        if self._loaded:
            raise ValueError("Cannot load a single object multiple times.")
        self._loaded = True
        return self._load(simulator)

    @abstractmethod
    def _load(self, simulator):
        pass

    def get_position(self):
        """Get object position in the format of Array[x, y, z]"""
        return self.get_position_orientation()[0]

    def get_orientation(self):
        """Get object orientation as a quaternion in the format of Array[x, y, z, w]"""
        return self.get_position_orientation()[1]

    def get_position_orientation(self):
        """Get object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        pos, orn = p.getBasePositionAndOrientation(self.body_id)
        return np.array(pos), np.array(orn)

    def set_position(self, pos):
        """Set object position in the format of Array[x, y, z]"""
        old_orn = self.get_orientation()
        self.set_position_orientation(pos, old_orn)

    def set_orientation(self, orn):
        """Set object orientation as a quaternion in the format of Array[x, y, z, w]"""
        old_pos = self.get_position()
        self.set_position_orientation(old_pos, orn)

    def set_position_orientation(self, pos, orn):
        """Set object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)
        self.move_constraints(pos, orn)

    @abstractmethod
    def move_constraints(self, pos, orn):
        pass

    def dump_state(self):
        return {"new_pos": self.new_pos, "new_orn": self.new_orn}

    def load_state(self, dump):
        self.move_constraints(dump["new_pos"], dump["new_orn"])


class BRBody(BRPart):
    """
    A simple ellipsoid representing the robot's body.
    """

    def __init__(self, parent, **kwargs):
        # Set up class
        self.parent = parent
        self.name = "BRBody_{}".format(self.parent.name)
        self.category = "agent"
        self.model = self.name
        self.movement_cid = None

        # Load in body from correct urdf, depending on user settings
        body_path = "normal_color" if self.parent.normal_color else "alternative_color"
        body_path_suffix = "vr_body.urdf" if not self.parent.use_tracked_body else "vr_body_tracker.urdf"
        self.vr_body_fpath = os.path.join(assets_path, "models", "vr_agent", "vr_body", body_path, body_path_suffix)
        super(BRBody, self).__init__(**kwargs)

    def _load(self, simulator):
        """
        Overidden load that keeps BRBody awake upon initialization.
        """
        self.body_id = p.loadURDF(self.vr_body_fpath, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.main_body = -1
        self.bounding_box = [0.5, 0.5, 1]
        self.mass = BODY_MASS  # p.getDynamicsInfo(body_id, -1)[0]
        # The actual body is at link 0, the base link is a "virtual" link
        p.changeDynamics(self.body_id, 0, mass=self.mass)
        p.changeDynamics(self.body_id, -1, mass=1e-9)

        simulator.load_object_in_renderer(
            self.parent, self.body_id, self.parent.class_id, **self.parent._rendering_params
        )

        self.movement_cid = p.createConstraint(
            self.body_id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            self.get_position(),
            [0, 0, 0, 1],
            self.get_orientation(),
        )

        return [self.body_id]

    def set_body_collision_filters(self):
        """
        Sets BRBody's collision filters.
        """
        no_col_groups = [SPECIAL_COLLISION_GROUPS[cat] for cat in BODY_NO_COLLISION_CATEGORIES]
        collision_mask = get_collision_group_mask(no_col_groups)

        body_link_idxs = [-1] + list(range(p.getNumJoints(self.body_id)))
        for body_link_idx in body_link_idxs:
            p.setCollisionFilterGroupMask(self.body_id, body_link_idx, self.parent.collision_group, collision_mask)

    def move_constraints(self, pos, orn):
        p.changeConstraint(self.movement_cid, pos, orn, maxForce=BODY_MOVING_FORCE)

    def command_position(self, action):
        """
        Updates BRBody to new position and rotation, via constraints.
        :param action: numpy array of actions.
        """
        # Compute the target world position from the delta position.
        delta_pos, delta_orn = action[:3], p.getQuaternionFromEuler(action[3:6])
        target_pos, target_orn = p.multiplyTransforms(*self.get_position_orientation(), delta_pos, delta_orn)

        # Clip the height.
        target_pos = [target_pos[0], target_pos[1], np.clip(target_pos[2], BODY_HEIGHT_RANGE[0], BODY_HEIGHT_RANGE[1])]

        self.new_pos = np.round(target_pos, 5).tolist()
        self.new_orn = np.round(target_orn, 5).tolist()

        self.move_constraints(self.new_pos, self.new_orn)

    def reset_position(self, reset_val):
        """
        Reset BRBody to new position and rotation, via teleportation.
        :param reset_val: numpy array of joint values to reset
        """
        # Compute the target world position from the delta position.
        delta_pos, delta_orn = reset_val[:3], p.getQuaternionFromEuler(reset_val[3:6])
        target_pos, target_orn = p.multiplyTransforms(*self.get_position_orientation(), delta_pos, delta_orn)

        # Clip the height.
        target_pos = [target_pos[0], target_pos[1], np.clip(target_pos[2], BODY_HEIGHT_RANGE[0], BODY_HEIGHT_RANGE[1])]

        self.new_pos = np.round(target_pos, 5).tolist()
        self.new_orn = np.round(target_orn, 5).tolist()

        self.set_position_orientation(self.new_pos, self.new_orn)

    def set_position_orientation(self, pos, orn):
        super(BRBody, self).set_position_orientation(pos, orn)
        self.new_pos = pos
        self.new_orn = orn

    def command_reset(self, val):
        if val > 0.5:  # The unnormalized action space for this button is 0 to 1. This thresholds that space into half.
            self.set_position_orientation(self.new_pos, self.new_orn)


class BRHand(BRPart):
    """
    Represents the human hand used for VR programs and robotics applications.
    """

    def __init__(self, parent, hand="right"):
        if hand not in ["left", "right"]:
            raise ValueError("ERROR: BRHand can only accept left or right as a hand argument!")

        hand_path = "normal_color" if parent.normal_color else "alternative_color"
        self.vr_hand_folder = os.path.join(assets_path, "models", "vr_agent", "vr_hand", hand_path)
        final_suffix = "vr_hand_{}.urdf".format(hand)

        self.parent = parent
        self.fpath = os.path.join(self.vr_hand_folder, final_suffix)
        self.hand = hand

        # Bool indicating whether the hands have been spwaned by pressing the trigger reset
        self.movement_cid = None
        self.name = "{}_hand_{}".format(self.hand, self.parent.name)
        self.model = self.name
        self.category = "agent"

        super(BRHand, self).__init__()

        # Keeps track of previous ghost hand hidden state
        self.prev_ghost_hand_hidden_state = False
        if self.parent.use_ghost_hands:
            self.ghost_hand = VisualMarker(
                visual_shape=p.GEOM_MESH,
                filename=os.path.join(
                    assets_path, "models", "vr_agent", "vr_hand", "ghost_hand_{}.obj".format(self.hand)
                ),
                scale=[0.001] * 3,
                class_id=self.parent.class_id,
            )
            self.ghost_hand.category = "agent"

    def _load(self, simulator):
        self.body_id = p.loadURDF(self.fpath, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        self.mass = p.getDynamicsInfo(self.body_id, -1)[0]

        simulator.load_object_in_renderer(
            self.parent, self.body_id, self.parent.class_id, **self.parent._rendering_params
        )

        body_ids = [self.body_id]
        if self.parent.use_ghost_hands:
            body_ids.extend(self.ghost_hand.load(simulator))
            self.ghost_hand.set_position_orientation(*self.get_position_orientation())
            p.changeVisualShape(self.ghost_hand.get_body_ids()[0], -1, rgbaColor=(0, 0, 0, 0))
            # change it to transparent for visualization

        p.changeDynamics(self.body_id, -1, mass=1, lateralFriction=HAND_FRICTION)
        for joint_index in range(p.getNumJoints(self.body_id)):
            # Make masses larger for greater stability
            # Mass is in kg, friction is coefficient
            p.changeDynamics(self.body_id, joint_index, mass=0.1, lateralFriction=HAND_FRICTION)

        # Create constraint that can be used to move the hand
        self.movement_cid = p.createConstraint(
            self.body_id,
            -1,
            -1,
            -1,
            p.JOINT_FIXED,
            [0, 0, 0],
            [0, 0, 0],
            self.get_position(),
            [0.0, 0.0, 0.0, 1.0],
            self.get_orientation(),
        )
        p.changeConstraint(self.movement_cid, maxForce=HAND_LIFTING_FORCE)

        return body_ids

    def set_position_orientation(self, pos, orn):
        # set position and orientation of BRobot body part and update
        # local transforms, note this function gets around state bound
        super(BRHand, self).set_position_orientation(pos, orn)
        self.new_pos = pos
        self.new_orn = orn
        # Update pos and orientation of ghost hands as well
        if self.parent.use_ghost_hands:
            self.ghost_hand.set_position_orientation(self.new_pos, self.new_orn)

    def get_local_position_orientation(self):
        # TODO: Deprecate in favor of link-based version.
        body = self.parent._parts["body"]
        return p.multiplyTransforms(
            *p.invertTransform(*body.get_position_orientation()), *self.get_position_orientation()
        )

    def command_position(self, action):
        """
        Updates position and close fraction of hand.
        :param action: numpy array of actions.
        """
        # These are relative to the shoulder.
        new_local_pos = action[0:3]
        new_local_orn = p.getQuaternionFromEuler(action[3:6])

        # Calculate new world position based on local transform and new body pose
        body = self.parent._parts["body"]
        shoulder = self.parent.links[self.hand + "_hand_shoulder"]
        new_shoulder_pos, new_shoulder_orn = p.multiplyTransforms(
            body.new_pos, body.new_orn, *shoulder.get_local_position_orientation()
        )
        self.new_pos, self.new_orn = p.multiplyTransforms(
            new_shoulder_pos, new_shoulder_orn, new_local_pos, new_local_orn
        )
        # Round to avoid numerical inaccuracies
        self.new_pos = np.round(self.new_pos, 5).tolist()
        self.new_orn = np.round(self.new_orn, 5).tolist()

        self.move_constraints(self.new_pos, self.new_orn)

        # Update ghost hands
        if self.parent.use_ghost_hands:
            self.update_ghost_hands()

    def reset_position(self, reset_val):
        """
        Reset BRHand to new position and rotation, via teleportation.
        :param reset_val: numpy array of joint values to reset
        """
        # These are relative to the shoulder.
        new_local_pos = reset_val[0:3]
        new_local_orn = p.getQuaternionFromEuler(reset_val[3:6])

        # Calculate new world position based on local transform and new body pose
        body = self.parent._parts["body"]
        shoulder = self.parent.links[self.hand + "_hand_shoulder"]
        new_shoulder_pos, new_shoulder_orn = p.multiplyTransforms(
            body.new_pos, body.new_orn, *shoulder.get_local_position_orientation()
        )
        self.new_pos, self.new_orn = p.multiplyTransforms(
            new_shoulder_pos, new_shoulder_orn, new_local_pos, new_local_orn
        )
        # Round to avoid numerical inaccuracies
        self.new_pos = np.round(self.new_pos, 5).tolist()
        self.new_orn = np.round(self.new_orn, 5).tolist()

        self.set_position_orientation(self.new_pos, self.new_orn)

    def command_reset(self, val):
        self.parent._parts["body"].command_reset(val)
        if val > 0.5:  # The unnormalized action space for this button is 0 to 1. This thresholds that space into half.
            self.set_position_orientation(self.new_pos, self.new_orn)

    def move_constraints(self, pos, orn):
        p.changeConstraint(self.movement_cid, pos, orn, maxForce=HAND_LIFTING_FORCE)

    def set_close_fraction(self, close_frac):
        """
        Sets the close fraction of the hand - this must be implemented by each subclass.
        """
        raise NotImplementedError()

    def update_ghost_hands(self):
        """
        Updates ghost hand to track real hand and displays it if the real and virtual hands are too far apart.
        """
        # Ghost hand tracks real hand whether it is hidden or not
        self.ghost_hand.set_position(self.new_pos)
        self.ghost_hand.set_orientation(self.new_orn)

        # If distance between hand and controller is greater than threshold,
        # ghost hand appears
        dist_to_real_controller = np.linalg.norm(np.array(self.new_pos) - np.array(self.get_position()))
        should_hide = dist_to_real_controller <= HAND_GHOST_HAND_APPEAR_THRESHOLD

        # Only toggle hidden state if we are transition from hidden to unhidden, or the other way around
        if not self.prev_ghost_hand_hidden_state and should_hide:
            self.parent.simulator.set_hidden_state(self.ghost_hand, hide=True)
            self.prev_ghost_hand_hidden_state = True
        elif self.prev_ghost_hand_hidden_state and not should_hide:
            self.parent.simulator.set_hidden_state(self.ghost_hand, hide=False)
            self.prev_ghost_hand_hidden_state = False


class BREye(BRPart):
    """
    Class representing the eye of the robot - robots can use this eye's position
    to move the camera and render the same thing that the VR users see.
    """

    def __init__(self, parent):
        # Set up class
        self.parent = parent

        self.name = "BREye_{}".format(self.parent.name)
        self.category = "agent"

        color_folder = "normal_color" if self.parent.normal_color else "alternative_color"
        self.head_visual_path = os.path.join(assets_path, "models", "vr_agent", "vr_eye", color_folder, "vr_head.obj")
        self.eye_path = os.path.join(assets_path, "models", "vr_agent", "vr_eye", "vr_eye.urdf")
        super(BREye, self).__init__()

        self.should_hide = True
        self.head_visual_marker = VisualMarker(
            visual_shape=p.GEOM_MESH, filename=self.head_visual_path, scale=[0.08] * 3, class_id=self.parent.class_id
        )
        self.neck_cid = None

    def _load(self, simulator):
        flags = p.URDF_USE_MATERIAL_COLORS_FROM_MTL | p.URDF_ENABLE_SLEEPING
        self.body_id = p.loadURDF(self.eye_path, flags=flags)

        # Set a minimal mass
        self.mass = 1e-9
        p.changeDynamics(self.body_id, -1, self.mass)

        simulator.load_object_in_renderer(
            self.parent, self.body_id, self.parent.class_id, **self.parent._rendering_params
        )

        body_ids = [self.body_id] + self.head_visual_marker.load(simulator)

        # Create a rigid constraint between the body and the head such that the head will move with the body during the
        # next physics simulation duration. Set the joint frame to be aligned with the child frame (URDF standard)
        local_pos, local_orn = self.get_local_position_orientation()
        self.neck_cid = p.createConstraint(
            parentBodyUniqueId=self.parent._parts["body"].body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=local_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=local_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )

        return body_ids

    def get_local_position_orientation(self):
        body = self.parent._parts["body"]
        return p.multiplyTransforms(
            *p.invertTransform(*body.get_position_orientation()), *self.get_position_orientation()
        )

    def set_position_orientation(self, pos, orn):
        # set position and orientation of BRobot body part and update
        # local transforms, note this function gets around state bound
        super(BREye, self).set_position_orientation(pos, orn)
        self.new_pos = pos
        self.new_orn = orn
        self.head_visual_marker.set_position_orientation(self.new_pos, self.new_orn)

    def move_constraints(self, pos, orn):
        body = self.parent._parts["body"]
        local_pos, local_orn = p.multiplyTransforms(*p.invertTransform(*body.get_position_orientation()), pos, orn)
        p.removeConstraint(self.neck_cid)
        self.neck_cid = p.createConstraint(
            parentBodyUniqueId=body.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=self.body_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=local_pos,
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=local_orn,
            childFrameOrientation=[0, 0, 0, 1],
        )

    def command_position(self, action):
        """
        Updates BREye to be where HMD is.
        :param action: numpy array of actions.
        """
        if not self.parent.show_visual_head and self.should_hide:
            self.parent.simulator.set_hidden_state(self.head_visual_marker, hide=True)
            self.should_hide = False

        # These are relative to the neck.
        new_local_pos = action[0:3]
        new_local_orn = p.getQuaternionFromEuler(action[3:6])

        # Calculate new world position based on new local transform and current body pose
        neck = self.parent.links["neck"]
        self.new_pos, self.new_orn = p.multiplyTransforms(
            *neck.get_position_orientation(), new_local_pos, new_local_orn
        )
        self.new_pos = np.round(self.new_pos, 5).tolist()
        self.new_orn = np.round(self.new_orn, 5).tolist()
        self.set_position_orientation(self.new_pos, self.new_orn)
