import os

import numpy as np
import pybullet as p

import igibson
from igibson.controllers import ControlType
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key

DEFAULT_ARM_POSES = {
    "vertical",
    "diagonal15",
    "diagonal30",
    "diagonal45",
    "horizontal",
}

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}


class Tiago(ManipulationRobot, TwoWheelRobot, ActiveCameraRobot):
    """
    Tiago Robot
    Reference: https://pal-robotics.com/robots/tiago/
    """

    def __init__(
        self,
        reset_joint_pos=None,
        rigid_trunk=False,
        default_trunk_offset=0.365,
        default_arm_pose="vertical",
        **kwargs,
    ):
        """
        :param reset_joint_pos: None or str or Array[float], if specified, should be the joint positions that the robot
            should be set to during a reset. If str, should be one of {tuck, untuck}, corresponds to default
            configurations for un/tucked modes. If None (default), self.default_joint_pos (untuck mode) will be used
            instead.
        :param rigid_trunk: bool, if True, will prevent the trunk from moving during execution.
        :param default_trunk_offset: float, sets the default height of the robot's trunk
        :param default_arm_pose: Default pose for the robot arm. Should be one of {"vertical", "diagonal15",
            "diagonal30", "diagonal45", "horizontal"}
        :param **kwargs: see ManipulationRobot, TwoWheelRobot, ActiveCameraRobot
        """
        # Store args
        self.rigid_trunk = rigid_trunk
        self.default_trunk_offset = default_trunk_offset
        assert_valid_key(key=default_arm_pose, valid_keys=DEFAULT_ARM_POSES, name="default_arm_pose")
        self.default_arm_pose = default_arm_pose

        # Parse reset joint pos if specifying special string
        if isinstance(reset_joint_pos, str):
            assert (
                reset_joint_pos in RESET_JOINT_OPTIONS
            ), "reset_joint_pos should be one of {} if using a string!".format(RESET_JOINT_OPTIONS)
            reset_joint_pos = (
                self.tucked_default_joint_pos if reset_joint_pos == "tuck" else self.untucked_default_joint_pos
            )

        # Run super init
        super().__init__(reset_joint_pos=reset_joint_pos, **kwargs)

    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "Tiago"

    @property
    def n_arms(self):
        return 2

    @property
    def arm_names(self):
        return ["left", "right"]

    @property
    def tucked_default_joint_pos(self):
        return np.array(
            [
                0.0,
                0.0,  # wheels
                0.0,  # trunk
                0.0,
                0.0,  # head
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # arm
                0.0,
                0.0,  # gripper
            ]
        )

    @property
    def untucked_default_joint_pos(self):
        pos = np.zeros(self.n_joints)
        pos[self.base_control_idx] = 0.0
        pos[self.trunk_control_idx] = 0.02 + self.default_trunk_offset
        pos[self.camera_control_idx] = np.array([0.0, 0.45])
        for arm in self.arm_names:
            pos[self.gripper_control_idx[arm]] = np.array([0.05, 0.05])  # open gripper

            # Choose arm based on setting
            if self.default_arm_pose == "vertical":
                pos[self.arm_control_idx[arm]] = np.array(
                    # [-0.94121, -0.64134, 1.55186, 1.65672, -0.93218, 1.53416, 2.14474]
                    [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
                )
            elif self.default_arm_pose == "diagonal15":
                pos[self.arm_control_idx[arm]] = np.array(
                    # [-0.95587, -0.34778, 1.46388, 1.47821, -0.93813, 1.4587, 1.9939]
                    [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
                )
            elif self.default_arm_pose == "diagonal30":
                pos[self.arm_control_idx[arm]] = np.array(
                    # [-1.06595, -0.22184, 1.53448, 1.46076, -0.84995, 1.36904, 1.90996]
                    [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
                )
            elif self.default_arm_pose == "diagonal45":
                pos[self.arm_control_idx[arm]] = np.array(
                    # [-1.11479, -0.0685, 1.5696, 1.37304, -0.74273, 1.3983, 1.79618]
                    [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
                )
            elif self.default_arm_pose == "horizontal":
                pos[self.arm_control_idx[arm]] = np.array(
                    # [-1.43016, 0.20965, 1.86816, 1.77576, -0.27289, 1.31715, 2.01226]
                    [0.22, 0.48, 1.52, 1.76, 0.04, -0.49, 0]
                )
            else:
                raise ValueError("Unknown default arm pose: {}".format(self.default_arm_pose))
        return pos

    def _create_discrete_action_space(self):
        # Tiago does not support discrete actions
        raise ValueError("Tiago does not support discrete actions!")

    def tuck(self):
        """
        Immediately set this robot's configuration to be in tucked mode
        """
        self.set_joint_positions(self.tucked_default_joint_pos)

    def untuck(self):
        """
        Immediately set this robot's configuration to be in untucked mode
        """
        self.set_joint_positions(self.untucked_default_joint_pos)

    def load(self, simulator):
        # Run super method
        ids = super().load(simulator)

        assert len(ids) == 1, "Tiago robot is expected to have only one body ID."

        # Extend super method by increasing laterial friction for EEF
        for arm in self.arm_names:
            for link in self.finger_joint_ids[arm]:
                p.changeDynamics(self.base_link.body_id, link, lateralFriction=500)

        return ids

    def _actions_to_control(self, action):
        # Run super method first
        u_vec, u_type_vec = super()._actions_to_control(action=action)

        # Override trunk value if we're keeping the trunk rigid
        if self.rigid_trunk:
            u_vec[self.trunk_control_idx] = self.untucked_default_joint_pos[self.trunk_control_idx]
            u_type_vec[self.trunk_control_idx] = ControlType.POSITION

        # Return control
        return u_vec, u_type_vec

    def _get_proprioception_dict(self):
        dic = super()._get_proprioception_dict()

        # Add trunk info
        dic["trunk_qpos"] = self.joint_positions[self.trunk_control_idx]
        dic["trunk_qvel"] = self.joint_velocities[self.trunk_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos"]

    @property
    def controller_order(self):
        controllers = ["base", "camera"]
        for arm in self.arm_names:
            controllers += ["arm_{}".format(arm), "gripper_{}".format(arm)]

        return controllers

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and IK controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        for arm in self.arm_names:
            controllers["arm_{}".format(arm)] = "InverseKinematicsController"
            controllers["gripper_{}".format(arm)] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        for arm in self.arm_names:
            # Use default IK controller -- also need to override joint idx being controlled to include trunk in default
            # IK arm controller
            cfg["arm_{}".format(arm)]["InverseKinematicsController"]["joint_idx"] = np.concatenate(
                [self.trunk_control_idx, self.arm_control_idx[self.default_arm]]
            )

            # If using rigid trunk, we also clamp its limits
            if self.rigid_trunk:
                cfg["arm_{}".format(arm)]["InverseKinematicsController"]["control_limits"]["position"][0][
                    self.trunk_control_idx
                ] = self.untucked_default_joint_pos[self.trunk_control_idx]
                cfg["arm_{}".format(arm)]["InverseKinematicsController"]["control_limits"]["position"][1][
                    self.trunk_control_idx
                ] = self.untucked_default_joint_pos[self.trunk_control_idx]

        return cfg

    @property
    def default_joint_pos(self):
        return self.untucked_default_joint_pos

    @property
    def wheel_radius(self):
        return 0.0613

    @property
    def wheel_axle_length(self):
        return 0.372

    @property
    def gripper_link_to_grasp_point(self):
        return {self.default_arm: np.array([0.1, 0, 0])}

    @property
    def assisted_grasp_start_points(self):
        return {
            arm: [
                GraspingPoint(link_name="gripper_{}_right_finger_link".format(arm), position=[0.04, -0.012, 0.014]),
                GraspingPoint(link_name="gripper_{}_right_finger_link".format(arm), position=[0.04, -0.012, -0.014]),
                GraspingPoint(link_name="gripper_{}_right_finger_link".format(arm), position=[-0.04, -0.012, 0.014]),
                GraspingPoint(link_name="gripper_{}_right_finger_link".format(arm), position=[-0.04, -0.012, -0.014]),
            ]
            for arm in self.arm_names
        }

    @property
    def assisted_grasp_end_points(self):
        return {
            arm: [
                GraspingPoint(link_name="gripper_{}_left_finger_link".format(arm), position=[0.04, 0.012, 0.014]),
                GraspingPoint(link_name="gripper_{}_left_finger_link".format(arm), position=[0.04, 0.012, -0.014]),
                GraspingPoint(link_name="gripper_{}_left_finger_link".format(arm), position=[-0.04, 0.012, 0.014]),
                GraspingPoint(link_name="gripper_{}_left_finger_link".format(arm), position=[-0.04, 0.012, -0.014]),
            ]
            for arm in self.arm_names
        }

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([1, 0])

    @property
    def trunk_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([2])

    @property
    def camera_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([4, 3])

    @property
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        return {"left": np.array([5, 6, 7, 8, 9, 10, 11]), "right": np.array([14, 15, 16, 17, 18, 19, 20])}

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        return {"left": np.array([12, 13]), "right": np.array([21, 22])}

    @property
    def disabled_collision_pairs(self):
        return [
            # ["torso_lift_link", "shoulder_lift_link"],
            # ["torso_lift_link", "torso_fixed_link"],
            # ["caster_wheel_link", "estop_link"],
            # ["caster_wheel_link", "laser_link"],
            # ["caster_wheel_link", "torso_fixed_link"],
            # ["caster_wheel_link", "l_wheel_link"],
            # ["caster_wheel_link", "r_wheel_link"],
        ]

    @property
    def eef_link_names(self):
        return {arm: "gripper_{}_link".format(arm) for arm in self.arm_names}

    @property
    def finger_link_names(self):
        return {
            arm: ["gripper_{}_right_finger_link".format(arm), "gripper_{}_left_finger_link".format(arm)]
            for arm in self.arm_names
        }

    @property
    def finger_joint_names(self):
        return {
            arm: ["gripper_{}_right_finger_joint".format(arm), "gripper_{}_left_finger_joint".format(arm)]
            for arm in self.arm_names
        }

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/tiago/tiago_dual.urdf")

    def dump_config(self):
        """Dump robot config"""
        dump = super(Tiago, self).dump_config()
        dump.update(
            {
                "rigid_trunk": self.rigid_trunk,
                "default_trunk_offset": self.default_trunk_offset,
                "default_arm_pose": self.default_arm_pose,
            }
        )
        return dump
