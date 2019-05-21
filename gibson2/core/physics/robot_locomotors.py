from gibson2.core.physics.robot_bases import BaseRobot
from gibson2.utils.utils import rotate_vector_3d
import numpy as np
import pybullet as p
import os
import gym, gym.spaces
from transforms3d.euler import euler2quat, euler2mat
from transforms3d.quaternions import quat2mat, qmult
import transforms3d.quaternions as quat
import sys


class WalkerBase(BaseRobot):
    """ Built on top of BaseRobot
    Handles action_dim, sensor_dim, scene
    base_position, apply_action, calc_state
    reward
    """

    def __init__(
            self,
            filename,    # robot file name
            robot_name,    # robot name
            action_dim,    # action dimension
            power,
            scale,
            sensor_dim=None,
            resolution=512,
            control='torque',
            is_discrete=True,
    ):
        BaseRobot.__init__(self, filename, robot_name, scale)
        self.control = control
        self.resolution = resolution
        self.is_discrete = is_discrete

        assert type(action_dim) == int, "Action dimension must be int, got {}".format(
            type(action_dim))
        self.action_dim = action_dim

        # deprecated
        self.obs_dim = [self.resolution, self.resolution, 0]
        obs_high = np.inf * np.ones(self.obs_dim)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)

        assert type(sensor_dim) == int, "Sensor dimension must be int, got {}".format(
            type(sensor_dim))
        self.sensor_dim = sensor_dim
        sensor_high = np.inf * np.ones([self.sensor_dim])
        self.sensor_space = gym.spaces.Box(-sensor_high, sensor_high, dtype=np.float32)

        if self.is_discrete:
            self.set_up_discrete_action_space()
        else:
            self.set_up_continuous_action_space()

        self.power = power
        self.camera_x = 0
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
        for j in self.ordered_joints:
            j.reset_joint_state(np.random.uniform(low=-0.1, high=0.1), 0)

    def get_position(self):
        '''Get current robot position
        '''
        return self.robot_body.get_position()

    def get_orientation(self):
        '''Return robot orientation
        '''
        return self.robot_body.get_orientation()

    def set_position(self, pos):
        self.robot_body.reset_position(pos)

    def set_orientation(self, orn):
        self.robot_body.set_orientation(orn)

    def move_by(self, delta):
        new_pos = np.array(delta) + self.get_position()
        self.robot_body.reset_position(new_pos)

    def move_forward(self, forward=0.05):
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([forward, 0, 0])))

    def move_backward(self, backward=0.05):
        x, y, z, w = self.robot_body.get_orientation()
        self.move_by(quat2mat([w, x, y, z]).dot(np.array([-backward, 0, 0])))

    def turn_left(self, delta=0.03):
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(-delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def turn_right(self, delta=0.03):
        orn = self.robot_body.get_orientation()
        new_orn = qmult((euler2quat(delta, 0, 0)), orn)
        self.robot_body.set_orientation(new_orn)

    def get_rpy(self):
        return self.robot_body.get_rpy()

    def apply_real_action(self, action):
        if self.control == 'torque':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_torque(self.power * j.power_coef * float(np.clip(action[n], -1, +1)))
        elif self.control == 'velocity':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_velocity(self.power * j.power_coef * float(np.clip(action[n], -1, +1)))
        elif self.control == 'position':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_position(action[n])
        elif type(self.control) is list or type(
                self.control) is tuple:    # if control is a tuple, set different control
            # type for each joint
            for n, j in enumerate(self.ordered_joints):
                if self.control[n] == 'torque':
                    j.set_motor_torque(self.power * j.power_coef *
                                       float(np.clip(action[n], -1, +1)))
                elif self.control[n] == 'velocity':
                    j.set_motor_velocity(self.power * j.power_coef *
                                         float(np.clip(action[n], -1, +1)))
                elif self.control[n] == 'position':
                    j.set_motor_position(action[n])
        else:
            pass

    def apply_action(self, action):
        if self.is_discrete:
            if isinstance(action, (list, np.ndarray)):
                assert len(action) == 1 and isinstance(action[0], (np.int64, int)), \
                    "discrete action has incorrect format"
                action = action[0]
            real_action = self.action_list[action]
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            real_action = self.action_high * action
        self.apply_real_action(real_action)

    def calc_state(self):
        j = np.array([j.get_joint_relative_state() for j in self.ordered_joints],
                     dtype=np.float32).flatten()
        self.joint_speeds = j[1::3]
        self.joint_torque = j[2::3]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::3]) > 0.99)

        z = self.robot_body.get_position()[2]
        r, p, yaw = self.robot_body.get_rpy()

        # rotate speed back to body point of view
        vx, vy, vz = rotate_vector_3d(self.robot_body.velocity(), r, p, yaw)
        more = np.array([z, vx, vy, vz, r, p, yaw], dtype=np.float32)

        return np.clip(np.concatenate([more] + [j]), -5, +5)


class Ant(WalkerBase):
    model_type = "MJCF"
    default_scale = 1

    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 1.0)
        WalkerBase.__init__(
            self,
            "ant.xml",
            "torso",
            action_dim=8,
            sensor_dim=28,
            power=2.5,
            scale=config.get("robot_scale", self.default_scale),
            resolution=config.get("resolution", 64),
            is_discrete=config.get("is_discrete", True),
            control="torque",
        )

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim, ),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        ## Hip_1, Ankle_1, Hip_2, Ankle_2, Hip_3, Ankle_3, Hip_4, Ankle_4
        self.action_list = [[self.torque, 0, 0, 0, 0, 0, 0, 0],
                            [0, self.torque, 0, 0, 0, 0, 0, 0],
                            [0, 0, self.torque, 0, 0, 0, 0, 0],
                            [0, 0, 0, self.torque, 0, 0, 0, 0],
                            [0, 0, 0, 0, self.torque, 0, 0, 0],
                            [0, 0, 0, 0, 0, self.torque, 0, 0],
                            [0, 0, 0, 0, 0, 0, self.torque, 0],
                            [0, 0, 0, 0, 0, 0, 0, self.torque],
                            [-self.torque, 0, 0, 0, 0, 0, 0, 0],
                            [0, -self.torque, 0, 0, 0, 0, 0, 0],
                            [0, 0, -self.torque, 0, 0, 0, 0, 0],
                            [0, 0, 0, -self.torque, 0, 0, 0, 0],
                            [0, 0, 0, 0, -self.torque, 0, 0, 0],
                            [0, 0, 0, 0, 0, -self.torque, 0, 0],
                            [0, 0, 0, 0, 0, 0, -self.torque, 0],
                            [0, 0, 0, 0, 0, 0, 0, -self.torque],
                            [0, 0, 0, 0, 0, 0, 0, 0]]  # yapf: disable
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('1'), ): 0,
            (ord('2'), ): 1,
            (ord('3'), ): 2,
            (ord('4'), ): 3,
            (ord('5'), ): 4,
            (ord('6'), ): 5,
            (ord('7'), ): 6,
            (ord('8'), ): 7,
            (ord('9'), ): 8,
            (ord('0'), ): 9,
            (ord('q'), ): 10,
            (ord('w'), ): 11,
            (ord('e'), ): 12,
            (ord('r'), ): 13,
            (ord('t'), ): 14,
            (ord('y'), ): 15,
            (): 4
        }


class Humanoid(WalkerBase):
    self_collision = True
    model_type = "MJCF"
    default_scale = 1
    glass_offset = 0.3

    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 0.1)
        self.glass_id = None
        WalkerBase.__init__(
            self,
            "humanoid.xml",
            "torso",
            action_dim=17,
            sensor_dim=40,
            power=0.41,
            scale=config.get("robot_scale", self.default_scale),
            resolution=config.get("resolution", 64),
            is_discrete=config.get("is_discrete", True),
            control="torque",
        )

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim, ),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.torque] * self.action_dim, [0.0] * self.action_dim]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def robot_specific_reset(self):
        humanoidId = -1
        numBodies = p.getNumBodies()
        for i in range(numBodies):
            bodyInfo = p.getBodyInfo(i)
            if bodyInfo[1].decode("ascii") == 'humanoid':
                humanoidId = i
        ## Spherical radiance/glass shield to protect the robot's camera

        WalkerBase.robot_specific_reset(self)

        if self.glass_id is None:
            glass_path = os.path.join(self.physics_model_dir, "glass.xml")
            glass_id = p.loadMJCF(glass_path)[0]
            self.glass_id = glass_id
            p.changeVisualShape(self.glass_id, -1, rgbaColor=[0, 0, 0, 0])
            p.createMultiBody(baseVisualShapeIndex=glass_id, baseCollisionShapeIndex=-1)
            cid = p.createConstraint(humanoidId,
                                     -1,
                                     self.glass_id,
                                     -1,
                                     p.JOINT_FIXED,
                                     jointAxis=[0, 0, 0],
                                     parentFramePosition=[0, 0, self.glass_offset],
                                     childFramePosition=[0, 0, 0])

        robot_pos = list(self.get_position())
        robot_pos[2] += self.glass_offset
        robot_orn = self.get_orientation()
        p.resetBasePositionAndOrientation(self.glass_id, robot_pos, robot_orn)

        self.motor_names = ["abdomen_z", "abdomen_y", "abdomen_x"]
        self.motor_power = [100, 100, 100]
        self.motor_names += ["right_hip_x", "right_hip_z", "right_hip_y", "right_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["left_hip_x", "left_hip_z", "left_hip_y", "left_knee"]
        self.motor_power += [100, 100, 300, 200]
        self.motor_names += ["right_shoulder1", "right_shoulder2", "right_elbow"]
        self.motor_power += [75, 75, 75]
        self.motor_names += ["left_shoulder1", "left_shoulder2", "left_elbow"]
        self.motor_power += [75, 75, 75]
        self.motors = [self.jdict[n] for n in self.motor_names]

    def apply_action(self, action):
        if self.is_discrete:
            if isinstance(action, (list, np.ndarray)):
                assert len(action) == 1 and isinstance(action[0], (np.int64, int)), \
                    "discrete action has incorrect format"
                action = action[0]
            real_action = self.action_list[action]
            self.apply_real_action(real_action)
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            real_action = self.action_high * action
            force_gain = 1
            for i, m, power in zip(range(17), self.motors, self.motor_power):
                m.set_motor_torque(float(force_gain * power * self.power * real_action[i]))

    def setup_keys_to_action(self):
        self.keys_to_action = {(ord('w'), ): 0, (): 1}


class Husky(WalkerBase):
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1

    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 0.03)
        WalkerBase.__init__(self,
                            "husky.urdf",
                            "base_link",
                            action_dim=4,
                            sensor_dim=17,
                            power=2.5,
                            scale=config.get("robot_scale", self.default_scale),
                            resolution=config.get("resolution", 64),
                            is_discrete=config.get("is_discrete", True),
                            control="torque")

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim, ),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.torque, self.torque, self.torque, self.torque],
                            [-self.torque, -self.torque, -self.torque, -self.torque],
                            [self.torque, -self.torque, self.torque, -self.torque],
                            [-self.torque, self.torque, -self.torque, self.torque], [0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def steering_cost(self, action):
        if not self.is_discrete:
            return 0
        if action == 2 or action == 3:
            return -0.1
        else:
            return 0

    def robot_specific_reset(self):
        WalkerBase.robot_specific_reset(self)

    def alive_bonus(self, z, pitch):
        top_xyz = self.parts["top_bumper_link"].get_position()
        bottom_xyz = self.parts["base_link"].get_position()
        alive = top_xyz[2] > bottom_xyz[2]
        return +1 if alive else -100    # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'), ): 0,    ## forward
            (ord('s'), ): 1,    ## backward
            (ord('d'), ): 2,    ## turn right
            (ord('a'), ): 3,    ## turn left
            (): 4
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)
        angular_velocity = self.robot_body.angular_velocity()
        return np.concatenate((base_state, np.array(angular_velocity)))


class Quadrotor(WalkerBase):
    model_type = "URDF"
    default_scale = 1
    mjcf_scaling = 1

    def __init__(self, config):
        self.config = config
        self.torque = config.get("torque", 0.02)
        WalkerBase.__init__(self,
                            "quadrotor.urdf",
                            "base_link",
                            action_dim=6,
                            sensor_dim=6,
                            power=2.5,
                            scale=config.get("robot_scale", self.default_scale),
                            resolution=config.get("resolution", 64),
                            is_discrete=config.get("is_discrete", True),
                            control="torque")

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim, ),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.torque * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.torque, 0, 0, 0, 0, 0], [-self.torque, 0, 0, 0, 0, 0],
                            [0, self.torque, 0, 0, 0, 0], [0, -self.torque, 0, 0, 0, 0],
                            [0, 0, self.torque, 0, 0, 0], [0, 0, -self.torque, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def apply_action(self, action):
        if self.is_discrete:
            if isinstance(action, (list, np.ndarray)):
                assert len(action) == 1 and isinstance(action[0], (np.int64, int)), \
                    "discrete action has incorrect format"
                action = action[0]
            real_action = self.action_list[action]
        else:
            action = np.clip(action, self.action_space.low, self.action_space.high)
            real_action = self.action_high * action

        p.setGravity(0, 0, 0)
        p.resetBaseVelocity(self.robot_ids[0], real_action[:3], real_action[3:])

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'), ): 0,    ## +x
            (ord('s'), ): 1,    ## -x
            (ord('d'), ): 2,    ## +y
            (ord('a'), ): 3,    ## -y
            (ord('z'), ): 4,    ## +z
            (ord('x'), ): 5,    ## -z
            (): 6
        }


class Turtlebot(WalkerBase):
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1

    def __init__(self, config):
        self.config = config
        self.velocity = config.get("velocity", 1.0)
        WalkerBase.__init__(self,
                            "turtlebot/turtlebot.urdf",
                            "base_link",
                            action_dim=2,
                            sensor_dim=16,
                            power=2.5,
                            scale=config.get("robot_scale", self.default_scale),
                            resolution=config.get("resolution", 64),
                            is_discrete=config.get("is_discrete", True),
                            control="velocity")

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim, ),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.velocity * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.velocity, self.velocity], [-self.velocity, -self.velocity],
                            [self.velocity * 0.5, -self.velocity * 0.5],
                            [-self.velocity * 0.5, self.velocity * 0.5], [0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'), ): 0,    # forward
            (ord('s'), ): 1,    # backward
            (ord('d'), ): 2,    # turn right
            (ord('a'), ): 3,    # turn left
            (): 4    # stay still
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)
        angular_velocity = self.robot_body.angular_velocity()
        return np.concatenate((base_state, np.array(angular_velocity)))


class JR2(WalkerBase):
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1

    def __init__(self, config):
        self.config = config
        self.velocity = config.get('velocity', 0.1)
        WalkerBase.__init__(self,
                            "jr2_urdf/jr2.urdf",
                            "base_link",
                            action_dim=4,
                            sensor_dim=17,
                            power=2.5,
                            scale=config.get("robot_scale", self.default_scale),
                            resolution=config.get("resolution", 64),
                            is_discrete=config.get("is_discrete", True),
                            control='velocity')

    def set_up_continuous_action_space(self):
        self.action_space = gym.spaces.Box(shape=(self.action_dim, ),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)
        self.action_high = self.velocity * np.ones([self.action_dim])
        self.action_low = -self.action_high

    def set_up_discrete_action_space(self):
        self.action_list = [[self.velocity, self.velocity, 0, self.velocity],
                            [-self.velocity, -self.velocity, 0, -self.velocity],
                            [self.velocity, -self.velocity, -self.velocity, 0],
                            [-self.velocity, self.velocity, self.velocity, 0], [0, 0, 0, 0]]
        self.action_space = gym.spaces.Discrete(len(self.action_list))
        self.setup_keys_to_action()

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'), ): 0,    ## forward
            (ord('s'), ): 1,    ## backward
            (ord('d'), ): 2,    ## turn right
            (ord('a'), ): 3,    ## turn left
            (): 4
        }

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)
        angular_velocity = self.robot_body.angular_velocity()
        return np.concatenate((base_state, np.array(angular_velocity)))


class JR2_Kinova(WalkerBase):
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1

    def __init__(self, config):
        '''
        idx: 1, name: left_wheel
        idx: 2, name: right_wheel
        idx: 15, name: pan_joint
        idx: 16, name: tilt_joint
        idx: 25, name: m1n6s200_joint_1
        idx: 26, name: m1n6s200_joint_2
        idx: 27, name: m1n6s200_joint_3
        idx: 28, name: m1n6s200_joint_4
        idx: 29, name: m1n6s200_joint_5
        idx: 30, name: m1n6s200_joint_6
        idx: 32, name: m1n6s200_joint_finger_1
        idx: 34, name: m1n6s200_joint_finger_2
        '''
        self.config = config
        self.wheel_velocity = config.get('wheel_velocity', 0.1)
        self.wheel_dim = 2
        self.cam_dim = 2
        self.arm_velocity = config.get('arm_velocity', 0.01)
        self.arm_dim = 8
        WalkerBase.__init__(self,
                            "jr2_urdf/jr2_kinova.urdf",
                            "base_link",
                            action_dim=12,
                            sensor_dim=46,
                            power=2.5,
                            scale=config.get("robot_scale", self.default_scale),
                            resolution=config.get("resolution", 64),
                            is_discrete=config.get("is_discrete", True),
                            control='velocity')

    def set_up_continuous_action_space(self):
        self.action_high = np.array([self.wheel_velocity] * 2 + [self.arm_velocity] * 8)
        self.action_low = -self.action_high
        self.action_space = gym.spaces.Box(shape=(self.wheel_dim + self.arm_dim, ),
                                           low=-1.0,
                                           high=1.0,
                                           dtype=np.float32)

    def set_up_discrete_action_space(self):
        assert False, "JR2_Kinova does not support discrete actions"

    def apply_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        denormalized_action = self.action_high * action
        real_action = np.zeros(self.action_dim)
        real_action[:self.wheel_dim] = denormalized_action[:self.wheel_dim]
        real_action[(self.wheel_dim + self.cam_dim):] = denormalized_action[self.wheel_dim:]
        self.apply_real_action(real_action)

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)
        angular_velocity = self.robot_body.angular_velocity()
        return np.concatenate((base_state, np.array(angular_velocity)))

    def get_end_effector_position(self):
        return self.parts['m1n6s200_link_finger_1'].get_position()