import gym
import numpy as np
import pybullet as p

from gibson2.robots.robot_locomotor import LocomotorRobot


class JR2_Kinova(LocomotorRobot):
    def __init__(self, config):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 0.3)
        self.wheel_dim = 2
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.arm_dim = 5

        LocomotorRobot.__init__(self,
                                "jr2_urdf/jr2_kinova.urdf",
                                action_dim=self.wheel_dim + self.arm_dim,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control='velocity',
                                self_collision=True)

    def set_up_continuous_action_space(self):
        self.action_high = np.array([self.wheel_velocity] * self.wheel_dim + [self.arm_velocity] * self.arm_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.wheel_dim + self.arm_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        assert False, "JR2_Kinova does not support discrete actions"

    def get_end_effector_position(self):
        return self.parts['m1n6s200_end_effector'].get_position()

    # initialize JR's arm to almost the same height as the door handle to ease exploration
    def robot_specific_reset(self):
        super(JR2_Kinova, self).robot_specific_reset()
        self.ordered_joints[2].reset_joint_state(-np.pi / 2.0, 0.0)
        self.ordered_joints[3].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[4].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[5].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[6].reset_joint_state(0.0, 0.0)

    def load(self):
        ids = super(JR2_Kinova, self).load()
        robot_id = self.robot_ids[0]

        # disable collision between base_chassis_joint and pan_joint
        #                   between base_chassis_joint and tilt_joint
        #                   between base_chassis_joint and camera_joint
        #                   between jr2_fixed_body_joint and pan_joint
        #                   between jr2_fixed_body_joint and tilt_joint
        #                   between jr2_fixed_body_joint and camera_joint
        p.setCollisionFilterPair(robot_id, robot_id, 0, 17, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 18, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 0, 19, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 1, 17, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 1, 18, 0)
        p.setCollisionFilterPair(robot_id, robot_id, 1, 19, 0)

        return ids