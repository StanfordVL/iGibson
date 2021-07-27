import gym
import numpy as np
import pybullet as p

from igibson.robots.robot_locomotor import LocomotorRobot
from igibson.external.pybullet_tools.utils import joints_from_names, set_joint_positions


class JR2_Kinova(LocomotorRobot):
    """
    JR2 Kinova robot
    Reference: https://cvgl.stanford.edu/projects/jackrabbot/
    Uses joint velocity control
    """

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
        """
        Set up continuous action space
        """
        self.action_high = np.array(
            [self.wheel_velocity] * self.wheel_dim + [self.arm_velocity] * self.arm_dim)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.wheel_dim + self.arm_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "JR2_Kinova does not support discrete actions"

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.parts['m1n6s200_end_effector'].get_position()

    def robot_specific_reset(self):
        """
        JR2 Kinova robot specific reset.
        Initialize JR's arm to about the same height at its neck, facing forward
        """
        super(JR2_Kinova, self).robot_specific_reset()
        self.ordered_joints[2].reset_joint_state(-np.pi / 2.0, 0.0)
        self.ordered_joints[3].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[4].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[5].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[6].reset_joint_state(0.0, 0.0)

    def load(self):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(JR2_Kinova, self).load()
        robot_id = self.robot_ids[0]

        disable_collision_names = [
            ['base_chassis_joint', 'pan_joint'],
            ['base_chassis_joint', 'tilt_joint'],
            ['base_chassis_joint', 'camera_joint'],
            ['jr2_fixed_body_joint', 'pan_joint'],
            ['jr2_fixed_body_joint', 'tilt_joint'],
            ['jr2_fixed_body_joint', 'camera_joint'],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        return ids
