import os

import numpy as np

import igibson
from igibson.external.pybullet_tools.utils import set_joint_positions
from igibson.robots.manipulation_robot import ManipulationRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.utils.constants import SemanticClass

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}


class JR2(ManipulationRobot, TwoWheelRobot):
    """
    JR2 Kinova robot
    Reference: https://cvgl.stanford.edu/projects/jackrabbot/
    """

    def __init__(self, reset_joint_pos=None, **kwargs):
        """
        :param reset_joint_pos: None or str or Array[float], if specified, should be the joint positions that the robot
            should be set to during a reset. If str, should be one of {tuck, untuck}, corresponds to default
            configurations for un/tucked modes. If None (default), self.default_joint_pos (untuck mode) will be used
            instead.
        :param **kwargs: see ManipulationRobot, TwoWheelRobot
        """
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
        return "JR2"

    def _create_discrete_action_space(self):
        # JR2 does not support discrete actions if we're controlling the arm as well
        raise ValueError("Full JR2 does not support discrete actions!")

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

    def reset(self):
        # In addition to normal reset, reset the joint configuration to be in default mode
        super().reset()
        joints = self.default_joint_pos
        set_joint_positions(self.get_body_ids()[0], [j.joint_id for j in self.joints.values()], joints)

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use differential drive with joint controller for arm, since arm only has 5DOF
        controllers["base"] = "DifferentialDriveController"
        controllers["arm_{}".format(self.default_arm)] = "JointController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_gripper_multi_finger_controller_configs(self):
        # Modify the default by inverting the command --> positive corresponds to gripper closed, not open!
        dic = super()._default_gripper_multi_finger_controller_configs

        for arm in self.arm_names:
            dic[arm]["inverted"] = True

        return dic

    @property
    def tucked_default_joint_pos(self):
        # todo: tune values
        return np.array([0.0, 0.0, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.0])

    @property
    def untucked_default_joint_pos(self):
        return np.array([0.0, 0.0, -np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2, 0.0, 0.0, 0.0, 0.0])

    @property
    def default_joint_pos(self):
        return self.untucked_default_joint_pos

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
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        return {self.default_arm: np.array([2, 3, 4, 5, 6, 7])}

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        return {self.default_arm: np.array([8, 9], dtype=int)}

    @property
    def disabled_collision_pairs(self):
        return [
            ["base_chassis_link", "pan_link"],
            ["base_chassis_link", "tilt_link"],
            ["base_chassis_link", "camera_link"],
            ["jr2_fixed_body_link", "pan_link"],
            ["jr2_fixed_body_link", "tilt_link"],
            ["jr2_fixed_body_link", "camera_link"],
        ]

    @property
    def eef_link_names(self):
        return {self.default_arm: "m1n6s200_end_effector"}

    @property
    def finger_link_names(self):
        return {self.default_arm: ["m1n6s200_link_finger_1", "m1n6s200_link_finger_2"]}

    @property
    def finger_joint_names(self):
        return {self.default_arm: ["m1n6s200_joint_finger_1", "m1n6s200_joint_finger_2"]}

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/jr2_urdf/jr2_kinova_gripper.urdf")
