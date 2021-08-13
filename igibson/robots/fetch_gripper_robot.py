import gym
import numpy as np
import pybullet as p
from IPython import embed

import igibson.utils.transform_utils as T
from igibson.controllers.ik_controller import IKController
from igibson.external.pybullet_tools.utils import (
    get_joint_info,
    get_relative_pose,
    joints_from_names,
    set_joint_positions,
)
from igibson.robots.robot_locomotor import LocomotorRobot


class FetchGripper(LocomotorRobot):
    """
    Fetch Robot
    Reference: https://fetchrobotics.com/robotics-platforms/fetch-mobile-manipulator/
    Uses joint velocity control
    """

    def __init__(self, config):
        self.config = config
        self.linear_velocity = config.get("linear_velocity", 1.0)  # m/s
        self.angular_velocity = config.get("angular_velocity", np.pi)  # rad/second
        self.head_velocity = config.get("head_velocity", 1.0)  # 1.0 represents maximum joint velocity
        self.arm_delta_pos_velocity = config.get("arm_delta_pos_velocity", 0.25)  # delta_pos = 0.25m
        self.arm_delta_orn_velocity = config.get("arm_delta_orn_velocity", np.deg2rad(30))  # delta_orn = 30deg
        self.gripper_velocity = config.get("gripper_velocity", 1.0)  # 1.0 represents maximum joint velocity
        self.default_arm_pose = config.get("default_arm_pose", "vertical")
        self.trunk_offset = config.get("trunk_offset", 0.0)
        self.wheel_dim = 2
        self.head_dim = 2
        self.arm_delta_pos_dim = 3
        self.arm_delta_orn_dim = 3
        self.gripper_dim = 1
        self.wheel_axle_half = 0.186  # half of the distance between the wheels
        self.wheel_radius = 0.0613  # radius of the wheels

        self.wheel_joint_ids = np.array([1, 2])
        self.head_joint_ids = np.array([4, 5])
        self.arm_joint_ids = np.array([3, 12, 13, 14, 15, 16, 17, 18])  # torso and arm
        self.gripper_joint_ids = np.array([20, 21])

        self.wheel_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.wheel_joint_ids]
        self.head_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.head_joint_ids]
        self.arm_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.arm_joint_ids]
        self.gripper_joint_action_idx = [i for i, idn in enumerate(self.joint_ids) if idn in self.gripper_joint_ids]

        LocomotorRobot.__init__(
            self,
            "fetch/fetch_gripper.urdf",
            action_dim=self.wheel_dim
            + self.head_dim
            + self.arm_delta_pos_dim
            + self.arm_delta_orn_dim
            + self.gripper_dim,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control=["differential_drive"] * 2 + ["velocity"] * 12,
            self_collision=False,
        )

    @property
    def joint_ids(self):
        return np.array([1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17, 18, 20, 21])

    @property
    def joint_damping(self):
        return np.array([get_joint_info(self.robot_ids[0], joint_id)[6] for joint_id in self.joint_ids])

    @property
    def num_joints(self):
        return len(self.joint_ids)

    @property
    def lower_joint_limits(self):
        return np.array(
            [
                -100.0,
                -100.0,
                0.0,
                -1.57,
                -0.76,
                -1.6056,
                -1.221,
                -100.0,
                -2.251,
                -100.0,
                -2.16,
                -100.0,
                0.0,
                0.0,
            ]
        )

    @property
    def upper_joint_limits(self):
        return np.array(
            [
                100.0,
                100.0,
                0.38615,
                1.57,
                1.45,
                1.6056,
                1.518,
                100.0,
                2.251,
                100.0,
                2.16,
                100.0,
                0.05,
                0.05,
            ]
        )

    @property
    def joint_range(self):
        return self.upper_joint_limits - self.lower_joint_limits

    @property
    def max_joint_velocities(self):
        return np.array(
            [
                17.4,
                17.4,
                0.1,
                1.57,
                1.57,
                1.256,
                1.454,
                1.571,
                1.521,
                1.571,
                2.268,
                2.268,
                0.05,
                0.05,
            ]
        )

    @property
    def eef_link_id(self):
        """
        Link corresponding to eef
        """
        return 19

    @property
    def tucked_default_joints(self):
        return np.array(
            [
                0.0,
                0.0,  # wheels
                0.02,  # trunk
                0.0,
                0.0,  # head
                1.1707963267948966,
                1.4707963267948965,
                -0.4,
                1.6707963267948966,
                0.0,
                1.5707963267948966,
                0.0,  # arm
                0.05,
                0.05,  # gripper
            ]
        )

    @property
    def untucked_default_joints(self):
        if self.default_arm_pose == "vertical":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -0.94121,
                    -0.64134,
                    1.55186,
                    1.65672,
                    -0.93218,
                    1.53416,
                    2.14474,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal15":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -0.95587,
                    -0.34778,
                    1.46388,
                    1.47821,
                    -0.93813,
                    1.4587,
                    1.9939,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal30":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.06595,
                    -0.22184,
                    1.53448,
                    1.46076,
                    -0.84995,
                    1.36904,
                    1.90996,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        elif self.default_arm_pose == "diagonal45":
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.11479,
                    -0.0685,
                    1.5696,
                    1.37304,
                    -0.74273,
                    1.3983,
                    1.79618,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )
        else:  # horizontal
            pose = np.array(
                [
                    0.0,
                    0.0,  # wheels
                    0.3 + self.trunk_offset,  # trunk
                    0.0,
                    0.45,  # head
                    -1.43016,
                    0.20965,
                    1.86816,
                    1.77576,
                    -0.27289,
                    1.31715,
                    2.01226,  # arm
                    0.05,
                    0.05,  # gripper
                ]
            )

        return pose

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_high = np.array(
            [self.linear_velocity]
            + [self.angular_velocity]
            + [self.head_velocity] * self.head_dim
            + [self.arm_delta_pos_velocity] * self.arm_delta_pos_dim
            + [self.arm_delta_orn_velocity] * self.arm_delta_orn_dim
            + [self.gripper_velocity] * self.gripper_dim
        )
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,), low=-1.0, high=1.0, dtype=np.float32)

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "Fetch does not support discrete actions"

    def robot_specific_reset(self):
        """
        Fetch robot specific reset.
        Reset the torso lift joint and tuck the arm towards the body
        """
        super(FetchGripper, self).robot_specific_reset()

        joints = self.untucked_default_joints
        set_joint_positions(self.robot_ids[0], self.joint_ids, joints)

        self.controller.reset()

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts["gripper_link"].get_position()

    def end_effector_part_index(self):
        """
        Get end-effector link id
        """
        return self.parts["gripper_link"].body_part_index

    def get_relative_eef_pose(self):
        """
        Get relative end-effector pose wrt robot base (returns 4x4 homogenous array)
        """
        return T.pose2mat(get_relative_pose(body=self.robot_ids[0], link1=self.eef_link_id))

    def get_relative_eef_position(self):
        """
        Get relative end-effector position wrt robot base
        """
        return self.get_relative_eef_pose()[:3, -1]

    def get_relative_eef_orientation(self):
        """
        Get relative end-effector orientation wrt robot base, in quaternion form
        """
        return T.mat2quat(self.get_relative_eef_pose()[:3, :3])

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(FetchGripper, self).load()
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ["torso_lift_joint", "shoulder_lift_joint"],
            ["torso_lift_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "estop_joint"],
            ["caster_wheel_joint", "laser_joint"],
            ["caster_wheel_joint", "torso_fixed_joint"],
            ["caster_wheel_joint", "l_wheel_joint"],
            ["caster_wheel_joint", "r_wheel_joint"],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        self.controller = IKController(robot=self, config=self.config)
        return ids

    def policy_action_to_robot_action(self, action):
        self.calc_state()
        # action has 2 + 2 + 6 + 1 = 11 dimensional
        robot_action = super(FetchGripper, self).policy_action_to_robot_action(action)
        new_robot_action = np.zeros(self.num_joints)

        # dim 0 and 1: linear and angular velocities of robot base
        new_robot_action[self.wheel_joint_action_idx] = robot_action[:2]

        # dim 2 and 3: head joint velocities
        new_robot_action[self.head_joint_action_idx] = robot_action[2:4]

        # dim 4-9: eef delta pos and orn
        new_robot_action[self.arm_joint_action_idx] = (
            self.controller.control(robot_action[4:10])[self.arm_joint_action_idx]
            / self.max_joint_velocities[self.arm_joint_action_idx]
        )

        # dim 10: gripper open/close
        new_robot_action[self.gripper_joint_action_idx] = robot_action[10]
        return new_robot_action
