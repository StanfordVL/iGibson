import os

import gym
import numpy as np
import pybullet as p

from igibson.robots.robot_locomotor import LocomotorRobot


class Humanoid(LocomotorRobot):
    """
    OpenAI Humanoid robot
    Uses joint torque control
    """

    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 0.1)
        self.glass_id = None
        self.glass_offset = 0.3
        LocomotorRobot.__init__(
            self,
            "humanoid/humanoid.xml",
            action_dim=17,
            torque_coef=0.41,
            scale=config.get("robot_scale", 1.0),
            is_discrete=config.get("is_discrete", False),
            control="torque",
            self_collision=True,
        )

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [[self.torque] *
                            self.action_dim, [0.0] * self.action_dim]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def robot_specific_reset(self):
        """
        Humanoid robot specific reset
        Add spherical radiance/glass shield to protect the robot's camera
        """
        humanoidId = -1
        numBodies = p.getNumBodies()
        for i in range(numBodies):
            bodyInfo = p.getBodyInfo(i)
            if bodyInfo[1].decode("ascii") == 'humanoid':
                humanoidId = i

        # Spherical radiance/glass shield to protect the robot's camera
        super(Humanoid, self).robot_specific_reset()

        if self.glass_id is None:
            glass_path = os.path.join(
                self.physics_model_dir, "humanoid/glass.xml")
            glass_id = p.loadMJCF(glass_path)[0]
            self.glass_id = glass_id
            p.changeVisualShape(self.glass_id, -1, rgbaColor=[0, 0, 0, 0])
            p.createMultiBody(baseVisualShapeIndex=glass_id,
                              baseCollisionShapeIndex=-1)
            cid = p.createConstraint(humanoidId,
                                     -1,
                                     self.glass_id,
                                     -1,
                                     p.JOINT_FIXED,
                                     jointAxis=[0, 0, 0],
                                     parentFramePosition=[
                                         0, 0, self.glass_offset],
                                     childFramePosition=[0, 0, 0])

        robot_pos = list(self.get_position())
        robot_pos[2] += self.glass_offset
        robot_orn = self.get_orientation()
        p.resetBasePositionAndOrientation(self.glass_id, robot_pos, robot_orn)

        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x",
                             "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x",
                             "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1",
                             "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

    def apply_action(self, action):
        """
        Apply policy action
        """
        real_action = self.policy_action_to_robot_action(action)
        if self.is_discrete:
            self.apply_robot_action(real_action)
        else:
            for i, m, joint_torque_coef in zip(range(17), self.motors, self.motor_power):
                m.set_motor_torque(
                    float(joint_torque_coef * self.torque_coef * real_action[i]))

    def setup_keys_to_action(self):
        self.keys_to_action = {(ord('w'),): 0, (): 1}
