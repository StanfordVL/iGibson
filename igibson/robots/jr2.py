import os

import numpy as np
import pybullet as p

import igibson
from igibson.external.pybullet_tools.utils import set_joint_positions
from igibson.robots.manipulation_robot import ManipulationRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils.constants import SemanticClass


class JR2(ManipulationRobot, TwoWheelRobot):
    """
    JR2 Kinova robot
    Reference: https://cvgl.stanford.edu/projects/jackrabbot/
    """

    def _create_discrete_action_space(self):
        # JR2 does not support discrete actions if we're controlling the arm as well
        raise ValueError("Full JR2 does not support discrete actions!")

    def _validate_configuration(self):
        # Make sure we're not using assisted grasping
        assert (
            self.assisted_grasping_mode is None
        ), "Cannot use assisted grasping modes for JR2 since gripper is disabled!"

        # Make sure we're using a null controller for the gripper
        assert (
            self.controller_config["gripper"]["name"] == "NullGripperController"
        ), "JR2 robot has its gripper disabled, so cannot use any controller other than NullGripperController!"

        # run super
        super()._validate_configuration()

    def reset(self):
        # In addition to normal reset, reset the joint configuration to be in default mode
        super().reset()
        joints = self.default_joint_pos
        set_joint_positions(self.get_body_id(), self.joint_ids, joints)

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "arm", "gripper"]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use differential drive with joint controller for arm, since arm only has 5DOF
        controllers["base"] = "DifferentialDriveController"
        controllers["arm"] = "JointController"
        controllers["gripper"] = "NullGripperController"

        return controllers

    @property
    def default_joint_pos(self):
        return np.array([0.0, 0.0, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0])

    @property
    def wheel_radius(self):
        return 0.2405

    @property
    def wheel_axle_length(self):
        return 0.5421

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([1, 0])

    @property
    def arm_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to arm joints.
        """
        return np.array([2, 3, 4, 5, 6])

    @property
    def gripper_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to gripper joints.
        """
        return np.array([], dtype=np.int)

    @property
    def disabled_collision_pairs(self):
        return [
            ["base_chassis_joint", "pan_joint"],
            ["base_chassis_joint", "tilt_joint"],
            ["base_chassis_joint", "camera_joint"],
            ["jr2_fixed_body_joint", "pan_joint"],
            ["jr2_fixed_body_joint", "tilt_joint"],
            ["jr2_fixed_body_joint", "camera_joint"],
        ]

    @property
    def eef_link_name(self):
        return "m1n6s200_end_effector"

    @property
    def finger_link_names(self):
        return []

    @property
    def finger_joint_names(self):
        return []

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/jr2_urdf/jr2_kinova.urdf")
