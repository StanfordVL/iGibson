import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from gibson2.robots.robot_locomotor import LocomotorRobot


class Fetch(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 1.0)
        self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        LocomotorRobot.__init__(self,
                                "fetch/fetch.urdf",
                                action_dim=self.wheel_dim + self.torso_lift_dim + self.arm_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity",
                                self_collision=True)

    def set_up_continuous_action_space(self):
        self.action_high = np.array([self.wheel_velocity] * self.wheel_dim +
                                    [self.torso_lift_velocity] * self.torso_lift_dim +
                                    [self.arm_velocity] * self.arm_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        assert False, "Fetch does not support discrete actions"

    def robot_specific_reset(self):
        super(Fetch, self).robot_specific_reset()

        # roll the arm to its body
        robot_id = self.robot_ids[0]
        arm_joints = joints_from_names(robot_id,
                                       [
                                           'torso_lift_joint',
                                           'shoulder_pan_joint',
                                           'shoulder_lift_joint',
                                           'upperarm_roll_joint',
                                           'elbow_flex_joint',
                                           'forearm_roll_joint',
                                           'wrist_flex_joint',
                                           'wrist_roll_joint'
                                       ])

        rest_position = (0.02, np.pi / 2.0 - 0.4, np.pi / 2.0 - 0.1, -0.4, np.pi / 2.0 + 0.1, 0.0, np.pi / 2.0, 0.0)
        # might be a better pose to initiate manipulation
        # rest_position = (0.30322468280792236, -1.414019864768982,
        #                  1.5178184935241699, 0.8189625336474915,
        #                  2.200358942909668, 2.9631312579803466,
        #                  -1.2862852996643066, 0.0008453550418615341)

        set_joint_positions(robot_id, arm_joints, rest_position)

    def get_end_effector_position(self):
        return self.parts['gripper_link'].get_position()

    def end_effector_part_index(self):
        return self.parts['gripper_link'].body_part_index

    def load(self):
        ids = super(Fetch, self).load()
        robot_id = self.robot_ids[0]

        # disable collision between torso_lift_joint and shoulder_lift_joint
        #                   between torso_lift_joint and torso_fixed_joint
        #                   between caster_wheel_joint and estop_joint
        #                   between caster_wheel_joint and laser_joint
        #                   between caster_wheel_joint and torso_fixed_joint
        #                   between caster_wheel_joint and l_wheel_joint
        #                   between caster_wheel_joint and r_wheel_joint
        p.setCollisionFilterPair(robot_id, robot_id, 3, 13, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 3, 22, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 20, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 21, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 22, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 1, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 2, 0)

        return ids