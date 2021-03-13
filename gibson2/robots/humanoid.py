import gym
import numpy as np
import pybullet as p
from gibson2.robots.robot_locomotor import LocomotorRobot
import time


class Humanoid_hri(LocomotorRobot):
    """
    Turtlebot robot
    Reference: http://wiki.ros.org/Robots/TurtleBot
    Uses joint velocity control
    """

    def __init__(self, config):
        self.config = config
        self.velocity = config.get("velocity", 1.0)
        self.knee = config.get("knee", False)

        if self.knee:
            file = "humanoid_hri/humanoid_hri_knee.urdf"
        else:
            file = "humanoid_hri/humanoid_hri.urdf"

        LocomotorRobot.__init__(self,
                                file,
                                action_dim=2,
                                scale=config.get("robot_scale", 1.0),
                                is_discrete=config.get("is_discrete", False),
                                control="velocity")

    def set_up_continuous_action_space(self):
        """
        Set up continuous action space
        """
        self.action_space = gym.spaces.Box(shape=(self.action_dim,),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = np.full(
            shape=self.action_dim, fill_value=self.velocity)
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        """
        Set up discrete action space
        """
        self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  # forward
            (ord('s'),): 1,  # backward
            (ord('d'),): 2,  # turn right
            (ord('a'),): 3,  # turn left
            (): 4  # stay still
        }

    def robot_specific_reset(self):
        """
        Robot specific reset. Apply zero velocity for all joints.
        """
        for j in self.ordered_joints:
            j.reset_joint_state(0.0, 0.0)

    def base_reset(self, pos):
        self.pelvis_id = 4
        self.base_cons = p.createConstraint(self.robot_ids[0], self.pelvis_id, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos, [0, 0, 0, 1]) # pelvis
        self.current_base_pos = np.array(pos)
        p.changeConstraint(self.base_cons, self.current_base_pos, maxForce=3000.0)

    def pose_reset(self, hand_pos, hand_ori):
        self.hand_id = 31
        self.current_hand_pos = np.array(hand_pos)
        self.current_hand_ori = np.array(hand_ori)

        goal_joints = [0.0 for i in range(p.getNumJoints(self.robot_ids[0]))]
        if self.knee:
            goal_joints[7] = -1.57
            goal_joints[9] = -1.57
            goal_joints[18] = -1.77
        p.setJointMotorControlArray(self.robot_ids[0], [i for i in range(p.getNumJoints(self.robot_ids[0]))], p.POSITION_CONTROL, goal_joints)

    def apply_action(self, action):
        delta_base_action = np.array(action[:3])
        delta_hand_action_pos = np.array(action[3:6])
        self.gripper = action[-1]

        self.current_base_pos += delta_base_action
        self.current_hand_pos += delta_hand_action_pos

        solutions = list(p.calculateInverseKinematics(self.robot_ids[0], self.hand_id, self.current_base_pos + self.current_hand_pos, self.current_hand_ori))[-11:]

        goal_joints = [0.0 for i in range(p.getNumJoints(self.robot_ids[0]))]
        if self.knee:
            goal_joints[7] = -1.57
            goal_joints[9] = -1.57
            goal_joints[18] = -1.77

        goal_joints[23] = solutions[0]
        goal_joints[24] = solutions[1]
        goal_joints[26] = solutions[2]
        goal_joints[28] = solutions[3]
        goal_joints[29] = solutions[4]
        goal_joints[30] = solutions[5]
        goal_joints[32] = self.gripper
        goal_joints[34] = self.gripper

        p.setJointMotorControlArray(self.robot_ids[0], [i for i in range(p.getNumJoints(self.robot_ids[0]))], p.POSITION_CONTROL, goal_joints)
        p.changeConstraint(self.base_cons, self.current_base_pos, maxForce = 3000.0)

        return
