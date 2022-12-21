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

# TODO: proprioceptive info

# We add support for both the mesh with r2d2 hand and the normal mesh
# Passive joint seems to have a lot of problem, so we stick with r2d2hand for now
r2d2_hand = True


class HSR(ManipulationRobot, TwoWheelRobot, ActiveCameraRobot):
    """
    HSRB Robot
    Reference: https://www.hsr.io/en/
    """

    def __init__(
        self,
        reset_joint_pos=None,
        rigid_trunk=False,
        **kwargs,
    ):
        """
        :param reset_joint_pos: None or str or Array[float], if specified, should be the joint positions that the robot
            should be set to during a reset. If str, should be one of {tuck, untuck}, corresponds to default
            configurations for un/tucked modes. If None (default), self.default_joint_pos (untuck mode) will be used
            instead.
        :param rigid_trunk: bool, if True, will prevent the trunk from moving during execution.
        :param default_trunk_offset: float, sets the default height of the robot's trunk
        :param **kwargs: see ManipulationRobot, TwoWheelRobot, ActiveCameraRobot
        """

        # Initialize link & joint dictionary

        if r2d2_hand:
            joint_list = ['base_roll_joint', 'base_r_drive_wheel_joint', 'base_l_drive_wheel_joint',
                          'base_r_passive_wheel_x_frame_joint', 'base_r_passive_wheel_y_frame_joint',
                          'base_r_passive_wheel_z_joint', 'base_l_passive_wheel_x_frame_joint',
                          'base_l_passive_wheel_y_frame_joint', 'base_l_passive_wheel_z_joint', 'torso_lift_joint',
                          'head_pan_joint', 'head_tilt_joint', 'arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint',
                          'wrist_flex_joint', 'wrist_roll_joint', 'left_gripper_joint', 'right_gripper_joint']

        else:
            joint_list = ['base_roll_joint', 'base_r_drive_wheel_joint', 'base_l_drive_wheel_joint',
                          'base_r_passive_wheel_x_frame_joint', 'base_r_passive_wheel_y_frame_joint',
                          'base_r_passive_wheel_z_joint', 'base_l_passive_wheel_x_frame_joint',
                          'base_l_passive_wheel_y_frame_joint', 'base_l_passive_wheel_z_joint', 'base_f_bumper_joint',
                          'base_b_bumper_joint', 'torso_lift_joint', 'head_pan_joint', 'head_tilt_joint',
                          'arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint',
                          'wrist_ft_sensor_frame_joint', 'hand_motor_joint', 'hand_l_proximal_joint',
                          'hand_l_spring_proximal_joint', 'hand_l_mimic_distal_joint', 'hand_l_distal_joint',
                          'hand_r_proximal_joint', 'hand_r_spring_proximal_joint', 'hand_r_mimic_distal_joint',
                          'hand_r_distal_joint']

        self.jn2i = {}
        for i in range(len(joint_list)):
            self.jn2i[joint_list[i]] = i

        # Store args
        self.rigid_trunk = rigid_trunk

        # Parse reset joint pos if specifying special string
        if isinstance(reset_joint_pos, str):
            raise Exception("reset joint pos str mode not implemented for HSR")

        # Run super init
        super().__init__(reset_joint_pos=reset_joint_pos, **kwargs)


    @property
    def model_name(self):
        """
        :return str: robot model name
        """
        return "HSR"

    def _create_discrete_action_space(self):
        # Fetch does not support discrete actions
        raise ValueError("HSR does not support discrete actions!")

    def load(self, simulator):
        # Run super method
        ids = super().load(simulator)

        assert len(ids) == 1, "HSR robot is expected to have only one body ID."

        # # Extend super method by increasing laterial friction for EEF
        # for link in self.finger_joint_ids[self.default_arm]:
        #     p.changeDynamics(self.base_link.body_id, link, lateralFriction=500)

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

        # # Add trunk info
        # dic["trunk_qpos"] = self.joint_positions[self.trunk_control_idx]
        # dic["trunk_qvel"] = self.joint_velocities[self.trunk_control_idx]

        return dic

    @property
    def default_proprio_obs(self):
        obs_keys = super().default_proprio_obs
        return obs_keys + ["trunk_qpos"]

    @property
    def controller_order(self):
        # Ordered by general robot kinematics chain
        return ["base", "camera", "arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and Joint controllers as default
        controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        # Use default IK controller -- also need to override joint idx being controlled to include trunk in default
        # IK arm controller
        cfg["arm_{}".format(self.default_arm)]["InverseKinematicsController"]["joint_idx"] = np.concatenate(
            [self.trunk_control_idx, self.arm_control_idx[self.default_arm]]
        )

        # If using rigid trunk, we also clamp its limits
        if self.rigid_trunk:
            cfg["arm_{}".format(self.default_arm)]["InverseKinematicsController"]["control_limits"]["position"][0][
                self.trunk_control_idx
            ] = self.untucked_default_joint_pos[self.trunk_control_idx]
            cfg["arm_{}".format(self.default_arm)]["InverseKinematicsController"]["control_limits"]["position"][1][
                self.trunk_control_idx
            ] = self.untucked_default_joint_pos[self.trunk_control_idx]

        return cfg

    @property
    def untucked_default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def default_joint_pos(self):
        return np.zeros(self.n_joints)

    @property
    def wheel_radius(self):
        # from the urdf
        return 0.04

    @property
    def wheel_axle_length(self):
        # TODO: this might not be accurate
        return 0.3

    @property
    def gripper_link_to_grasp_point(self):
        # Todo: need change
        return {self.default_arm: np.array([0.1, 0, 0])}

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        return np.array([self.jn2i["base_r_drive_wheel_joint"], self.jn2i["base_l_drive_wheel_joint"]])

    @property
    def trunk_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to trunk joint.
        """
        return np.array([self.jn2i['torso_lift_joint']])

    @property
    def camera_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [tilt, pan] camera joints.
        """
        return np.array([self.jn2i['head_pan_joint'], self.jn2i['head_tilt_joint']])

    @property
    def arm_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to arm joints.
        """
        arm_joint_names = ['arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint', 'wrist_flex_joint', 'wrist_roll_joint']
        return {self.default_arm: np.array([self.jn2i[joint_name] for joint_name in arm_joint_names])}

    @property
    def gripper_control_idx(self):
        """
        :return dict[str, Array[int]]: Dictionary mapping arm appendage name to indices in low-level control
            vector corresponding to gripper joints.
        """
        if r2d2_hand:
            return {self.default_arm: np.array([self.jn2i['left_gripper_joint'], self.jn2i['right_gripper_joint']])}
        else:
            return {self.default_arm: np.array([self.jn2i['hand_motor_joint']])}


    @property
    def disabled_collision_pairs(self):
        if r2d2_hand:
            return [
                ['wrist_roll_link', 'right_gripper'],
                ['wrist_roll_link', 'left_gripper'],
                ['wrist_flex_link', 'gripper_pole'],
                ['arm_roll_link', 'gripper_pole'],
                ['base_link', 'arm_flex_link'],
                ['base_link', 'base_l_passive_wheel_z_link'],
                ['base_link', 'base_r_passive_wheel_z_link'],
                ['base_link', 'base_l_drive_wheel_link'],
                ['base_link', 'base_r_drive_wheel_link']
            ]
        else:
            return [
                ['hand_r_spring_proximal_link', 'hand_r_distal_link'],
                ['hand_l_spring_proximal_link', 'hand_l_distal_link'],
                ['hand_r_spring_proximal_link', 'hand_palm_link'],
                ['hand_l_spring_proximal_link', 'hand_palm_link'],
                ['wrist_roll_link', 'hand_r_spring_proximal_link'],
                ['wrist_roll_link', 'hand_l_spring_proximal_link'],
                ['wrist_roll_link', 'hand_r_distal_link'],
                ['wrist_roll_link', 'hand_l_distal_link'],
                ['arm_roll_link', 'hand_r_distal_link'],
                ['arm_roll_link', 'hand_l_distal_link'],
                ['hand_r_distal_link', 'wrist_flex_link'],
                ['hand_l_distal_link', 'wrist_flex_link'],
                ['base_link', 'arm_flex_link'],
                ['base_link', 'base_l_passive_wheel_z_link'],
                ['base_link', 'base_r_passive_wheel_z_link'],
                ['base_link', 'base_l_drive_wheel_link'],
                ['base_link', 'base_r_drive_wheel_link'],
            ]

    @property
    def eef_link_names(self):
        if r2d2_hand:
            return {self.default_arm: "gripper_pole"}
        else:
            return {self.default_arm: "hand_palm_link"}

    @property
    def finger_link_names(self):
        if r2d2_hand:
            return {self.default_arm: ["right_tip", "left_tip"]}
        else:
            return {self.default_arm: ["hand_l_finger_tip_frame", "hand_r_finger_tip_frame"]}

    @property
    def finger_joint_names(self):
        if r2d2_hand:
            return {self.default_arm: ["left_gripper_joint", "right_gripper_joint"]}
        else:
            return {self.default_arm: ["hand_l_proximal_joint", "hand_r_proximal_joint"]}

    @property
    def model_file(self):
        if r2d2_hand:
            return os.path.join(igibson.assets_path, "models/hsr/hsrb4s.urdf")
        else:
            return os.path.join(igibson.assets_path, "models/hsr/hsrb4s.obj.urdf")

    def dump_config(self):
        """Dump robot config"""
        dump = super(HSR, self).dump_config()
        return dump
