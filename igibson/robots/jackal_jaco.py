import os

import numpy as np

import igibson
from igibson.external.pybullet_tools.utils import set_joint_positions
from igibson.robots.manipulation_robot import ManipulationRobot
from igibson.robots.locomotion_robot import LocomotionRobot
from igibson.utils.constants import SemanticClass

RESET_JOINT_OPTIONS = {
    "tuck",
    "untuck",
}


class JackalJaco(ManipulationRobot, LocomotionRobot):
    """
    Jackal Jaco robot
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
        return "JACKAL_JACO"

    def _create_discrete_action_space(self):
        # JACKAL_JACO does not support discrete actions if we're controlling the arm as well
        raise ValueError("JACKAL_JACO does not support discrete actions!")

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
            dic[arm]["inverted"] = False

        return dic

    @property
    def tucked_default_joint_pos(self):
        # todo: tune values
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.9, 1.3, 4.2, 1.4, 0.0])

    @property
    def untucked_default_joint_pos(self):
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 2.9, 1.3, 4.2, 1.4, 0.0])

    @property
    def default_joint_pos(self):
        return self.untucked_default_joint_pos

    @property
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        return {self.default_arm: np.array([4, 5, 6, 7, 8, 9])}

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        pass
        return {self.default_arm: np.array([10, 11], dtype=int)}

    @property
    def disabled_collision_pairs(self):
        return [
            ["chassis_link", "d435_cam1camera_link"],
        ]

    @property
    def eef_link_names(self):
        return {self.default_arm: "j2n6s300_end_effector"}

    @property
    def finger_link_names(self):
        return {self.default_arm: ["j2n6s300_link_finger_1", "j2n6s300_link_finger_2"]}

    @property
    def finger_joint_names(self):
        return {self.default_arm: ["j2n6s300_joint_finger_1", "j2n6s300_joint_finger_2"]}
    
    ##### LOCOMOTION #####
    @property
    def base_control_idx(self):
        return np.array([0, 1, 2, 3])

    # @property
    # def default_joint_pos(self):
    #     return np.zeros(self.n_joints)

    @property
    def model_file(self):
        return os.path.join(igibson.assets_path, "models/jackal_jaco/jackal_jaco.urdf")
