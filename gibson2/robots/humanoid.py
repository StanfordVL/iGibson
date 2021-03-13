import gym
import numpy as np
import pybullet as p
from gibson2.robots.robot_locomotor import LocomotorRobot


class Humanoid_hri(LocomotorRobot):
    """
    Turtlebot robot
    Reference: http://wiki.ros.org/Robots/TurtleBot
    Uses joint velocity control
    """

    def __init__(self, config):
        self.config = config
        self.velocity = config.get("velocity", 1.0)
        LocomotorRobot.__init__(self,
                                "humanoid_hri/humanoid_hri.urdf",
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

        self.base_cons = p.createConstraint(self.robot_ids[0], 4, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], pos, [0, 0, 0, 1]) # pelvis
        self.current_base_pos = pos

        self.current_hand_pos = [0.3932078205093461, -0.1700037605678417, 1.6820246348771142]
        self.current_hand_ori = [0.237, 0.0, 0.4744, 0.8478]
        self.gripper = 0.0

    def apply_robot_action(self, action):
        solutions = list(p.calculateInverseKinematics(self.robot_ids[0], 31, self.current_hand_pos, self.current_hand_ori))
        print(len(solutions))

        goal_joints = [0.0 for i in range(p.getNumJoints(self.robot_ids[0]))]
        goal_joints[23] = solutions[0]
        goal_joints[24] = solutions[1]
        goal_joints[26] = solutions[2]
        goal_joints[28] = solutions[3]
        goal_joints[29] = solutions[4]
        goal_joints[30] = solutions[5]
        goal_joints[32] = self.gripper
        goal_joints[34] = self.gripper

        p.setJointMotorControlArray(self.robot_ids[0], [i for i in range(p.getNumJoints(self.robot_ids[0]))], p.POSITION_CONTROL, goal_joints)

        self.current_hand_pos[-1] += 0.01
        if self.current_hand_pos[-1] > 1.9:
            self.current_hand_pos[-1] = 1.2
        self.gripper += 0.1
        if self.gripper > 1.0:
            self.gripper = 0.0

        self.current_base_pos[-1] += 0.01
        if self.current_base_pos[-1] > 1.9:
            self.current_base_pos[-1] = 1.2

        p.changeConstraint(self.base_cons, self.current_base_pos)

        return
