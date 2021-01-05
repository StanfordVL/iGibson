import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from gibson2.robots.robot_locomotor import LocomotorRobot


class FetchGripper(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        # self.wheel_velocity = config.get('wheel_velocity', 1.0)
        # self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        # self.arm_velocity = config.get('arm_velocity', 1.0)
        # self.gripper_velocity = config.get('gripper_velocity', 1.0)
        # self.head_velocity = config.get('gripper_velocity', 1.0)
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        self.gripper_dim = 2
        self.head_dim = 2
        action_dim = self.wheel_dim + self.torso_lift_dim + self.head_dim + self.arm_dim + self.gripper_dim
        self.max_velocity = np.array(config.get('max_velocity', np.ones(action_dim)))
        self.wheel_axle_half = 0.18738  # half of the distance between the wheels
        self.wheel_radius = 0.065  # radius of the wheels
        LocomotorRobot.__init__(self,
                                "fetch/fetch_gripper.urdf",
                                action_dim=action_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity",
                                self_collision=True)

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        # self.action_high = np.array([self.wheel_velocity] * self.wheel_dim +
        #                             [self.torso_lift_velocity] * self.torso_lift_dim +
        #                             [self.head_velocity] * self.head_dim +
        #                             [self.arm_velocity] * self.arm_dim +
        #                             [self.gripper_velocity] * self.gripper_dim)
        #self.action_high = np.array(self.max_velocity)
        #self.action_low = -self.action_high
        self.action_high = np.ones(self.action_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

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
        # roll the arm to its body
        robot_id = self.robot_ids[0]
        joints = joints_from_names(robot_id,
                                       [
                                           'torso_lift_joint',
                                           'head_pan_joint',
                                           'head_tilt_joint',
                                           'shoulder_pan_joint',
                                           'shoulder_lift_joint',
                                           'upperarm_roll_joint',
                                           'elbow_flex_joint',
                                           'forearm_roll_joint',
                                           'wrist_flex_joint',
                                           'wrist_roll_joint',
                                           'r_gripper_finger_joint',
                                           'l_gripper_finger_joint'
                                       ])
        rest_position = (0.02, 0., 0., np.pi / 2.0 - 0.4, np.pi / 2.0 - 0.1, -0.4, np.pi / 2.0 + 0.1, 0.0, np.pi / 2.0, 0.0, 0.05, 0.05)
        # might be a better pose to initiate manipulation
        # rest_position = (0.30322468280792236, -1.414019864768982,
        #                  1.5178184935241699, 0.8189625336474915,
        #                  2.200358942909668, 2.9631312579803466,
        #                  -1.2862852996643066, 0.0008453550418615341)
        set_joint_positions(robot_id, joints, rest_position)

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts['gripper_link'].get_position()

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(FetchGripper, self).load()
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ['torso_lift_joint', 'shoulder_lift_joint'],
            ['torso_lift_joint', 'torso_fixed_joint'],
            ['caster_wheel_joint', 'estop_joint'],
            ['caster_wheel_joint', 'laser_joint'],
            ['caster_wheel_joint', 'torso_fixed_joint'],
            ['caster_wheel_joint', 'l_wheel_joint'],
            ['caster_wheel_joint', 'r_wheel_joint'],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        return ids
