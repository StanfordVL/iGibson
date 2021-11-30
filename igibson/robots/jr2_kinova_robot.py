import gym
import numpy as np
import pybullet as p

from igibson.external.pybullet_tools.utils import joints_from_names
from igibson.robots.locomotion_robot import LocomotionRobot


class JR2_Kinova(LocomotionRobot):
    """
    JR2 Kinova robot
    Reference: https://cvgl.stanford.edu/projects/jackrabbot/
    Uses joint velocity control
    """

    def __init__(self, config, **kwargs):
        self.config = config
        self.wheel_dim = 2
        self.arm_dim = 5

        LocomotionRobot.__init__(
            self,
            "jr2_urdf/jr2_kinova.urdf",
            action_dim=self.wheel_dim + self.arm_dim,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="velocity",
            self_collision=True,
            **kwargs
        )

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        assert False, "JR2_Kinova does not support discrete actions"

    def get_end_effector_position(self):
        """
        Get end-effector position
        """
        return self.links["m1n6s200_end_effector"].get_position()

    def reset(self):
        """
        JR2 Kinova robot specific reset.
        Initialize JR's arm to about the same height at its neck, facing forward
        """
        super(JR2_Kinova, self).reset()
        self.ordered_joints[2].reset_joint_state(-np.pi / 2.0, 0.0)
        self.ordered_joints[3].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[4].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[5].reset_joint_state(np.pi / 2.0, 0.0)
        self.ordered_joints[6].reset_joint_state(0.0, 0.0)

    def load(self, simulator):
        """
        Load the robot into pybullet. Filter out unnecessary self collision
        due to modeling imperfection in the URDF
        """
        ids = super(JR2_Kinova, self).load(simulator)
        robot_id = self.get_body_id()

        disable_collision_names = [
            ["base_chassis_joint", "pan_joint"],
            ["base_chassis_joint", "tilt_joint"],
            ["base_chassis_joint", "camera_joint"],
            ["jr2_fixed_body_joint", "pan_joint"],
            ["jr2_fixed_body_joint", "tilt_joint"],
            ["jr2_fixed_body_joint", "camera_joint"],
        ]
        for names in disable_collision_names:
            link_a, link_b = joints_from_names(robot_id, names)
            p.setCollisionFilterPair(robot_id, robot_id, link_a, link_b, 0)

        return ids
