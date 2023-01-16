import os

import numpy as np
import pybullet as p

import igibson
from igibson.controllers import ControlType
from igibson.robots.active_camera_robot import ActiveCameraRobot
from igibson.robots.manipulation_robot import GraspingPoint, ManipulationRobot
from igibson.robots.locomotion_robot import LocomotionRobot
from igibson.robots.two_wheel_robot import TwoWheelRobot
from igibson.robots.robot_base import VirtualJoint, VirtualPlanarJoint
from igibson.utils.constants import SemanticClass
from igibson.utils.python_utils import assert_valid_key

BODY_LINEAR_VELOCITY = 0.05  # linear velocity thresholds in meters/frame
BODY_ANGULAR_VELOCITY = 0.05  # angular velocity thresholds in radians/frame
BODY_MOVING_FORCE = 500
EPSILON = 0.001
WHEEL_FRICTION = 0.001

# We add support for both the mesh with r2d2 hand and the normal mesh
# Passive joint seems to have a lot of problem, so we stick with r2d2hand for now
r2d2_hand = True
assert r2d2_hand == True, "We only support r2d2 hand"

# A switch between differential drive and omni_base
omni_base = True

if omni_base:
    loco_robot_class = LocomotionRobot
else:
    loco_robot_class = TwoWheelRobot


class HSR(ManipulationRobot, loco_robot_class, ActiveCameraRobot):
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

        if omni_base:
            joint_list = ['base_roll_joint', 'base_r_drive_wheel_joint', 'base_l_drive_wheel_joint', 'torso_lift_joint',
                          'head_pan_joint', 'head_tilt_joint', 'arm_lift_joint', 'arm_flex_joint', 'arm_roll_joint',
                          'wrist_flex_joint', 'wrist_roll_joint', 'left_gripper_joint', 'right_gripper_joint',
                          'body__body_x', 'body__body_y', 'body__body_z', 'body__body_rx', 'body__body_ry',
                          'body__body_rz']
        else:
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



        self.new_pos = [0, 0, 0]
        self.new_orn = [0, 0, 0, 1]
        self.movement_cid = None

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

        import ipdb
        ipdb.set_trace()
        # # Extend super method by increasing laterial friction for EEF
        # for link in self.finger_joint_ids[self.default_arm]:
        #     p.changeDynamics(self.base_link.body_id, link, lateralFriction=500)

        for i in range(p.getNumJoints(self.base_link.body_id)):
            if "wheel" in str(p.getJointInfo(self.base_link.body_id, i)[12]):
                p.changeDynamics(
                    self.base_link.body_id,
                    i,
                    lateralFriction=WHEEL_FRICTION,
                    spinningFriction=WHEEL_FRICTION,
                    rollingFriction=WHEEL_FRICTION,
                )

        return ids

    def _actions_to_control(self, action):
        # Run super method first
        u_vec, u_type_vec = super()._actions_to_control(action=action)

        # Hacky implementation of coupled joint
        u_vec[self.trunk_control_idx] = u_vec[self.jn2i['arm_lift_joint']] / 2.0
        u_type_vec[self.trunk_control_idx] = u_type_vec[self.jn2i['arm_lift_joint']]

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
        # Ordered by general robot kinematics chain
        return ["base", "camera", "arm_{}".format(self.default_arm), "gripper_{}".format(self.default_arm)]

    ###########################  Added from Tiago (for omni base)   ########################

    if omni_base:
        def _setup_virtual_joints(self):
            """
            Sets up the virtual joints for the omnidirectional base.
            """
            virtual_joints = []
            virtual_joints.extend(
                VirtualPlanarJoint(
                    joint_name="body__body",
                    parent_link=self.base_link,
                    child_link=self.base_link,
                    command_callback=self.base_command_position,
                    reset_callback=self.base_reset_position,
                    lower_limits=[None, None, None, None, None, None],
                    upper_limits=[None, None, None, None, None, None],
                ).get_joints()
            )
            return virtual_joints

    def move_constraints(self, pos, orn):
        """
        Updates the position and orientation of the PyBullet constraint corresponding to the omnidirectional base.
        :param pos: new position
        :param orn: new orientation
        """
        if self.movement_cid:
            p.changeConstraint(self.movement_cid, pos, orn, maxForce=BODY_MOVING_FORCE)
        else:
            self.movement_cid = p.createConstraint(
                parentBodyUniqueId=self.base_link.body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=-1,
                childLinkIndex=-1,
                jointType=p.JOINT_FIXED,
                jointAxis=[0, 0, 1],
                parentFramePosition=[0, 0, 0],
                childFramePosition=self.get_position(),
                parentFrameOrientation=[0, 0, 0, 1],
                childFrameOrientation=self.get_orientation(),
            )

    def base_command_position(self, action):
        """
        Updates Tiago to new position and rotation, via constraints.
        :param action: numpy array of actions.
        """
        # Compute the target world position from the delta position.
        vel_norm = np.linalg.norm(action[:3])
        clipped_vel_norm = min(vel_norm, BODY_LINEAR_VELOCITY)
        for i in range(3):
            action[i] = (clipped_vel_norm + EPSILON) / (vel_norm + EPSILON) * action[i]
        action[3:] = np.clip(action[3:], -BODY_ANGULAR_VELOCITY, BODY_ANGULAR_VELOCITY)
        delta_pos, delta_orn = action[:3], p.getQuaternionFromEuler(action[3:6])
        target_pos, target_orn = p.multiplyTransforms(*self.get_position_orientation(), delta_pos, delta_orn)

        self.new_pos = np.round(target_pos, 5).tolist()
        self.new_orn = np.round(target_orn, 5).tolist()

        self.move_constraints(self.new_pos, self.new_orn)

    def base_reset_position(self, reset_val):
        """
        Reset Tiago to new position and rotation, via teleportation.
        :param reset_val: numpy array of joint values to reset
        """
        # Compute the target world position from the delta position.
        delta_pos, delta_orn = reset_val[:3], p.getQuaternionFromEuler(reset_val[3:6])
        target_pos, target_orn = p.multiplyTransforms(*self.get_position_orientation(), delta_pos, delta_orn)

        # TODO: why is the target not used?
        # Clip the height.
        # target_pos = [target_pos[0], target_pos[1], np.clip(target_pos[2], BODY_HEIGHT_RANGE[0], BODY_HEIGHT_RANGE[1])]

        # self.new_pos = np.round(self.new_pos, 5).tolist()
        # self.new_orn = np.round(self.new_orn, 5).tolist()

        self.new_pos = np.round(target_pos, 5).tolist()
        self.new_orn = np.round(target_orn, 5).tolist()

        self.set_position_orientation(self.new_pos, self.new_orn)

    def get_position_orientation(self):
        """Get object position and orientation in the format of Tuple[Array[x, y, z], Array[x, y, z, w]]"""
        pos, orn = p.getBasePositionAndOrientation(self.base_link.body_id)
        return np.array(pos), np.array(orn)

    def set_position_orientation(self, pos, orn):
        # super(Tiago, self).set_position_orientation(pos, orn)
        self.new_pos = pos
        self.new_orn = orn
        p.resetBasePositionAndOrientation(self.base_link.body_id, pos, orn)
        self.move_constraints(pos, orn)

    def reset(self):
        # Move the constraint for each part to the default position.
        super().reset()
        self.set_position_orientation(*self.get_position_orientation())

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
                np.array([-BODY_LINEAR_VELOCITY] * 3 + [-BODY_ANGULAR_VELOCITY] * 3),
                np.array([BODY_LINEAR_VELOCITY] * 3 + [BODY_ANGULAR_VELOCITY] * 3),
            ),
            "command_output_limits": None,
        }
        return dic

    ###########################  End of Added from Tiago   ########################

    @property
    def _default_controllers(self):
        # Always call super first
        controllers = super()._default_controllers

        # We use multi finger gripper, differential drive, and Joint controllers as default
        if omni_base:
            controllers["base"] = "JointController"
        else:
            controllers["base"] = "DifferentialDriveController"
        controllers["camera"] = "JointController"
        controllers["arm_{}".format(self.default_arm)] = "InverseKinematicsController"
        controllers["gripper_{}".format(self.default_arm)] = "MultiFingerGripperController"

        return controllers

    @property
    def _default_controller_config(self):
        # Grab defaults from super method first
        cfg = super()._default_controller_config

        if omni_base:
            # Get default base controller for omnidirectional Tiago
            cfg["base"] = {"JointController": self._default_base_controller_configs}
        return cfg

    def hsr_joint_position(self):
        default = np.zeros(self.n_joints)
        default[self.jn2i["arm_lift_joint"]] = 0.07
        default[self.jn2i["torso_lift_joint"]] = 0.035
        default[self.jn2i["arm_flex_joint"]] = -1.57
        return default

    @property
    def untucked_default_joint_pos(self):
        return self.hsr_joint_position()

    @property
    def default_joint_pos(self):
        return self.hsr_joint_position()

    @property
    def wheel_radius(self):
        # Obtained from the urdf
        return 0.04

    @property
    def wheel_axle_length(self):
        # Obtained from HSR paper
        return 0.266  # 0.3

    @property
    def gripper_link_to_grasp_point(self):
        # Todo: need change, fetch is [0.1, 0, 0]
        return {self.default_arm: np.array([0, 0, 0])}

    @property
    def base_control_idx(self):
        """
        :return Array[int]: Indices in low-level control vector corresponding to [Left, Right] wheel joints.
        """
        if omni_base:
            joints = list(self.joints.keys())
            return np.array(
                [
                    joints.index("body__body_%s" % (component))
                    for component in VirtualPlanarJoint.ACTUATED_COMPONENT_SUFFIXES
                ]
            )
        else:
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
    def gripper_link_names(self):
        return {"gripper_pole", "left_gripper", "left_tip", "right_gripper", "right_tip"}

    @property
    def base_link_names(self):
        return {"base_footprint", "base_link", "base_roll_link", "base_r_drive_wheel_link", "base_l_drive_wheel_link",
                "base_r_passive_wheel_x_frame", "base_r_passive_wheel_y_frame", "base_r_passive_wheel_z_link",
                "base_l_passive_wheel_x_frame", "base_l_passive_wheel_y_frame", "base_l_passive_wheel_z_link",
                "base_range_sensor_link", "base_imu_frame", "torso_lift_link"}

    @property
    def head_link_names(self):
        return {"head_pan_link", "head_tilt_link", "head_l_stereo_camera_link", "head_l_stereo_camera_gazebo_frame",
                "head_r_stereo_camera_link", "head_r_stereo_camera_gazebo_frame", "head_center_camera_frame", "eyes",
                "head_center_camera_gazebo_frame", "head_rgbd_sensor_link", "head_rgbd_sensor_gazebo_frame"}

    @property
    def arm_link_names(self):
        return {"arm_lift_link", "arm_flex_link", "arm_roll_link", "wrist_flex_link",
                "wrist_roll_link", "wrist_ft_sensor_frame"}

    def get_link_group(self, link_name):
        if link_name in self.gripper_link_names:
            return "gripper"
        if link_name in self.base_link_names:
            return "base"
        if link_name in self.head_link_names:
            return "head"
        if link_name in self.arm_link_names:
            return "arm"
        raise NotImplementedError

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
