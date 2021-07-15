from igibson.robots.robot_base import BaseRobot
from igibson.utils.utils import rotate_vector_3d
import numpy as np
from transforms3d.euler import euler2quat
from transforms3d.quaternions import quat2mat, qmult


class LocomotorRobot(BaseRobot):
    """
    Built on top of BaseRobot.
    Provides common interface for a wide variety of robots.
    """

    def __init__(
            self,
            filename,  # robot file name
            action_dim,  # action dimension
            base_name=None,
            scale=1.0,
            control='torque',
            is_discrete=True,
            torque_coef=1.0,
            velocity_coef=1.0,
            self_collision=False
    ):
        BaseRobot.__init__(self, filename, base_name, scale, self_collision)
        self.control = control
        self.is_discrete = is_discrete

        assert type(action_dim) == int, "Action dimension must be int, got {}".format(
            type(action_dim))
        self.action_dim = action_dim

        if self.is_discrete:
            self.set_up_discrete_action_space()
        else:
            self.set_up_continuous_action_space()

        self.torque_coef = torque_coef
        self.velocity_coef = velocity_coef
        self.scale = scale

    def set_up_continuous_action_space(self):
        """
        Each subclass implements their own continuous action spaces
        """
        raise NotImplementedError

    def set_up_discrete_action_space(self):
        """
        Each subclass implements their own discrete action spaces
        """
        raise NotImplementedError

    def robot_specific_reset(self):
        """
        Robot specific reset. Apply zero velocity for all joints.
        """
        for j in self.ordered_joints:
            j.reset_joint_state(0.0, 0.0)

    def get_position(self):
        """
        Get current robot position
        """
        return self.robot_body.get_position()

    def get_orientation(self):
        """
        Return robot orientation

        :return: quaternion in xyzw
        """
        return self.robot_body.get_orientation()

    def get_rpy(self):
        """
        Return robot orientation in roll, pitch, yaw

        :return: roll, pitch, yaw
        """
        return self.robot_body.get_rpy()

    def set_position(self, pos):
        """
        Set robot position

        :param pos: robot position
        """
        self.robot_body.set_position(pos)

    def set_orientation(self, orn):
        """
        Set robot orientation

        :param orn: robot orientation
        """
        self.robot_body.set_orientation(orn)

    def set_position_orientation(self, pos, orn):
        """
        Set robot position and orientation

        :param pos: robot position
        :param orn: robot orientation
        """
        self.robot_body.set_pose(pos, orn)

    def get_linear_velocity(self):
        """
        Set robot base linear velocity

        :return: base linear velocity
        """
        return self.robot_body.get_linear_velocity()

    def get_angular_velocity(self):
        """
        Set robot base angular velocity

        :return: base angular velocity
        """
        return self.robot_body.get_angular_velocity()

    def move_by(self, delta):
        """
        Move robot base without physics simulation

        :param delta: delta base position
        """
        new_pos = np.array(delta) + self.get_position()
        self.robot_body.reset_position(new_pos)

    def move_forward(self, forward=0.05):
        """
        Move robot base forward without physics simulation

        :param forward: delta base position forward
        """
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([forward, 0, 0])))

    def move_backward(self, backward=0.05):
        """
        Move robot base backward without physics simulation

        :param backward: delta base position backward
        """
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([-backward, 0, 0])))

    def turn_left(self, delta=0.03):
        """
        Rotate robot base left without physics simulation

        :param delta: delta angle to rotate the base left
        """
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(-delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def turn_right(self, delta=0.03):
        """
        Rotate robot base right without physics simulation

        :param delta: delta angle to rotate the base right
        """
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def keep_still(self):
        """
        Keep the robot still. Apply zero velocity to all joints.
        """
        for n, j in enumerate(self.ordered_joints):
            j.set_motor_velocity(0.0)

    def apply_robot_action(self, action):
        """
        Apply robot action.
        Support joint torque, velocity, position control and
        differentiable drive / twist command control

        :param action: robot action
        """
        if self.control == 'torque':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_torque(
                    self.torque_coef * j.max_torque * float(np.clip(action[n], -1, +1)))
        elif self.control == 'velocity':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_velocity(
                    self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
        elif self.control == 'position':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_position(action[n])
        elif self.control == 'differential_drive':
            # assume self.ordered_joints = [left_wheel, right_wheel]
            assert action.shape[0] == 2 and len(
                self.ordered_joints) == 2, 'differential drive requires the first two joints to be two wheels'
            lin_vel, ang_vel = action
            if not hasattr(self, 'wheel_axle_half') or not hasattr(self, 'wheel_radius'):
                raise Exception(
                    'Trying to use differential drive, but wheel_axle_half and wheel_radius are not specified.')
            left_wheel_ang_vel = (lin_vel - ang_vel *
                                  self.wheel_axle_half) / self.wheel_radius
            right_wheel_ang_vel = (lin_vel + ang_vel *
                                   self.wheel_axle_half) / self.wheel_radius
            self.ordered_joints[0].set_motor_velocity(left_wheel_ang_vel)
            self.ordered_joints[1].set_motor_velocity(right_wheel_ang_vel)
        elif type(self.control) is list or type(self.control) is tuple:
            # if control is a tuple, set different control type for each joint
            if 'differential_drive' in self.control:
                # Assume the first two joints are wheels using differntiable drive control, and the rest use joint control
                # assume self.ordered_joints = [left_wheel, right_wheel, joint_1, joint_2, ...]
                assert action.shape[0] >= 2 and len(
                    self.ordered_joints) >= 2, 'differential drive requires the first two joints to be two wheels'
                assert self.control[0] == self.control[1] == 'differential_drive', 'differential drive requires the first two joints to be two wheels'
                lin_vel, ang_vel = action[:2]
                if not hasattr(self, 'wheel_axle_half') or not hasattr(self, 'wheel_radius'):
                    raise Exception(
                        'Trying to use differential drive, but wheel_axle_half and wheel_radius are not specified.')
                left_wheel_ang_vel = (
                    lin_vel - ang_vel * self.wheel_axle_half) / self.wheel_radius
                right_wheel_ang_vel = (
                    lin_vel + ang_vel * self.wheel_axle_half) / self.wheel_radius
                self.ordered_joints[0].set_motor_velocity(left_wheel_ang_vel)
                self.ordered_joints[1].set_motor_velocity(right_wheel_ang_vel)

            for n, j in enumerate(self.ordered_joints):
                if self.control[n] == 'torque':
                    j.set_motor_torque(
                        self.torque_coef * j.max_torque * float(np.clip(action[n], -1, +1)))
                elif self.control[n] == 'velocity':
                    j.set_motor_velocity(
                        self.velocity_coef * j.max_velocity * float(np.clip(action[n], -1, +1)))
                elif self.control[n] == 'position':
                    j.set_motor_position(action[n])
        else:
            raise Exception('unknown control type: {}'.format(self.control))

    def policy_action_to_robot_action(self, action):
        """
        Scale the policy action (always in [-1, 1]) to robot action based on action range

        :param action: policy action
        :return: robot action
        """
        if self.is_discrete:
            if isinstance(action, (list, np.ndarray)):
                assert len(action) == 1 and isinstance(action[0], (np.int64, int)), \
                    "discrete action has incorrect format"
                action = action[0]
            real_action = self.action_list[action]
        else:
            # self.action_space should always be [-1, 1] for policy training
            action = np.clip(action, self.action_space.low,
                             self.action_space.high)

            # de-normalize action to the appropriate, robot-specific scale
            real_action = (self.action_high - self.action_low) / 2.0 * action + \
                          (self.action_high + self.action_low) / 2.0
        return real_action

    def apply_action(self, action):
        """
        Apply policy action
        """
        real_action = self.policy_action_to_robot_action(action)
        self.apply_robot_action(real_action)

    def calc_state(self):
        """
        Calculate commonly used proprioceptive states for the robot

        :return: proprioceptive states
        """
        j = np.array([j.get_joint_relative_state()
                      for j in self.ordered_joints]).astype(np.float32).flatten()
        self.joint_position = j[0::3]
        self.joint_velocity = j[1::3]
        self.joint_torque = j[2::3]
        self.joint_at_limit = np.count_nonzero(
            np.abs(self.joint_position) > 0.99)

        pos = self.get_position()
        rpy = self.get_rpy()

        # rotate linear and angular velocities to local frame
        lin_vel = rotate_vector_3d(self.get_linear_velocity(), *rpy)
        ang_vel = rotate_vector_3d(self.get_angular_velocity(), *rpy)

        state = np.concatenate([pos, rpy, lin_vel, ang_vel, j])
        return state
