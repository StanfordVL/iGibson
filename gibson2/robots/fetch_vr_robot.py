import gym
import numpy as np
import pybullet as p

from gibson2.external.pybullet_tools.utils import joints_from_names, set_joint_positions
from gibson2.robots.robot_locomotor import LocomotorRobot


class FetchVR(LocomotorRobot):
    """
    Fetch robot used in VR embodiment demos.
    """
    def __init__(self, config):
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 1.0)
        self.torso_lift_velocity = config.get('torso_lift_velocity', 1.0)
        self.arm_velocity = config.get('arm_velocity', 1.0)
        self.wheel_dim = 2
        self.torso_lift_dim = 1
        self.arm_dim = 7
        LocomotorRobot.__init__(self,
                                "fetch/fetch_vr.urdf",
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

    def robot_specific_reset(self):
        super(FetchVR, self).robot_specific_reset()

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
    
    # Return body id of fetch robot
    def get_fetch_body_id(self):
        return self.robot_body.bodies[self.robot_body.body_index]

    def set_z_rotation(self, hmd_rot):
        """
        Sets the z rotation of the fetch VR robot using the provided HMD rotation.
        """
        # Get z component of hmd rotation
        _, _, hmd_z = p.getEulerFromQuaternion(hmd_rot)
        prev_x, prev_y, _ = p.getEulerFromQuaternion(self.get_orientation())
        # Preserve pre-existing x and y rotations, just force z rotation to be same as HMD
        fetch_rot = p.getQuaternionFromEuler([prev_x, prev_y, hmd_z])
        self.set_orientation(fetch_rot)

    # Set open/close fraction of the end grippers
    def set_fetch_gripper_fraction(self, frac, maxForce=500):
        min_joint = 0.0
        max_joint = 0.05
        right_finger_joint_idx = 20
        left_finger_joint_idx = 21
        # TODO: Set more friction on grippers using p.changeDynamics?
        #  min_joint + frac * (max_joint - min_joint)
        target_pos = 0.05
        p.setJointMotorControl2(self.get_fetch_body_id(),
                                right_finger_joint_idx, 
                                p.POSITION_CONTROL, 
                                targetPosition=target_pos, 
                                force=maxForce)
        
        p.setJointMotorControl2(self.get_fetch_body_id(),
                                left_finger_joint_idx, 
                                p.POSITION_CONTROL, 
                                targetPosition=target_pos, 
                                force=maxForce)

    def get_end_effector_position(self):
        return self.parts['gripper_link'].get_position()

    def load(self):
        ids = super(FetchVR, self).load()
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