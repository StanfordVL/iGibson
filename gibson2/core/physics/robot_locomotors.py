from gibson2.core.physics.robot_bases import BaseRobot
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

    def __init__(self,
                 filename,  # robot file name
                 robot_name,  # robot name
                 action_dim,  # action dimension
                 power,
                 initial_pos,
                 scale,
                 sensor_dim=None,
                 resolution=512,
                 control='torque',
                 ):
        BaseRobot.__init__(self, filename, robot_name, scale)
        self.control = control
        self.resolution = resolution
        self.obs_dim = None
        self.obs_dim = [self.resolution, self.resolution, 0]

        assert type(sensor_dim) == int, "Sensor dimension must be int, got {}".format(type(sensor_dim))
        assert type(action_dim) == int, "Action dimension must be int, got {}".format(type(action_dim))

        action_high = np.ones([action_dim])
        self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)
        obs_high = np.inf * np.ones(self.obs_dim)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high, dtype=np.float32)
        sensor_high = np.inf * np.ones([sensor_dim])
        self.sensor_space = gym.spaces.Box(-sensor_high, sensor_high, dtype=np.float32)

        self.power = power
        self.camera_x = 0
        self.initial_pos = initial_pos
        self.body_xyz = [0, 0, 0]
        self.action_dim = action_dim
        self.scale = scale
        self.sensor_dim = sensor_dim

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
        return self.robot_body.bp_pose.rpy()

    def apply_action(self, a):
        if isinstance(a, list):
            action = np.array(a)

        if self.control == 'torque':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
        elif self.control == 'velocity':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_velocity(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
        elif self.control == 'position':
            for n, j in enumerate(self.ordered_joints):
                j.set_motor_position(a[n])
        elif type(self.control) is list or type(self.control) is tuple:  # if control is a tuple, set different control
            # type for each joint
            for n, j in enumerate(self.ordered_joints):
                if self.control[n] == 'torque':
                    j.set_motor_torque(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
                elif self.control[n] == 'velocity':
                    j.set_motor_velocity(self.power * j.power_coef * float(np.clip(a[n], -1, +1)))
                elif self.control[n] == 'position':
                    j.set_motor_position(a[n])
        else:
            pass

    def calc_state(self):
        j = np.array([j.get_joint_relative_state() for j in self.ordered_joints], dtype=np.float32).flatten()
        self.joint_speeds = j[1::2]
        self.joints_at_limit = np.count_nonzero(np.abs(j[0::2]) > 0.99)

        body_pose = self.robot_body.pose()
        parts_xyz = np.array([p.pose().xyz() for p in self.parts.values()]).flatten()
        self.body_xyz = (
            parts_xyz[0::3].mean(), parts_xyz[1::3].mean(),
            body_pose.xyz()[2])  # torso z is more informative than mean z
        self.dist_to_start = np.linalg.norm(np.array(self.body_xyz) - np.array(self.initial_pos))
        self.body_rpy = body_pose.rpy()
        z = self.body_xyz[2]
        r, p, yaw = self.body_rpy
        rot_x = np.array(
            [[1, 0, 0],
             [0, np.cos(-r), -np.sin(-r)],
             [0, np.sin(-r), np.cos(-r)]]
        )
        rot_y = np.array(
            [[np.cos(-p), 0, np.sin(-p)],
             [0, 1, 0],
             [-np.sin(-p), 0, np.cos(-p)]]
        )
        rot_z = np.array(
            [[np.cos(-yaw), -np.sin(-yaw), 0],
             [np.sin(-yaw), np.cos(-yaw), 0],
             [0, 0, 1]]
        )
        # rotate speed back to body point of view
        vx, vy, vz = np.dot(rot_x, np.dot(rot_y, np.dot(rot_z, self.robot_body.velocity())))

        more = np.array([z,
                         0.3 * vx, 0.3 * vy, 0.3 * vz,
                         # 0.3 is just scaling typical speed into -1..+1, no physical sense here
                         r, p, yaw], dtype=np.float32)

        return np.clip(np.concatenate([more] + [j]), -5, +5)

class Ant(WalkerBase):
    model_type = "MJCF"
    default_scale = 1
    def __init__(self, config):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "ant.xml", "torso", action_dim=8,
                            sensor_dim=28, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            resolution=config["resolution"],
                            )
        self.r_f = 0.1
        self.is_discrete = config["is_discrete"]

        if config["is_discrete"]:
            self.action_space = gym.spaces.Discrete(17)
            self.torque = 10
            ## Hip_1, Ankle_1, Hip_2, Ankle_2, Hip_3, Ankle_3, Hip_4, Ankle_4 
            self.action_list = [[self.r_f * self.torque, 0, 0, 0, 0, 0, 0, 0],
                                [0, self.r_f * self.torque, 0, 0, 0, 0, 0, 0],
                                [0, 0, self.r_f * self.torque, 0, 0, 0, 0, 0],
                                [0, 0, 0, self.r_f * self.torque, 0, 0, 0, 0],
                                [0, 0, 0, 0, self.r_f * self.torque, 0, 0, 0],
                                [0, 0, 0, 0, 0, self.r_f * self.torque, 0, 0],
                                [0, 0, 0, 0, 0, 0, self.r_f * self.torque, 0],
                                [0, 0, 0, 0, 0, 0, 0, self.r_f * self.torque],
                                [-self.r_f * self.torque, 0, 0, 0, 0, 0, 0, 0],
                                [0, -self.r_f * self.torque, 0, 0, 0, 0, 0, 0],
                                [0, 0, -self.r_f * self.torque, 0, 0, 0, 0, 0],
                                [0, 0, 0, -self.r_f * self.torque, 0, 0, 0, 0],
                                [0, 0, 0, 0, -self.r_f * self.torque, 0, 0, 0],
                                [0, 0, 0, 0, 0, -self.r_f * self.torque, 0, 0],
                                [0, 0, 0, 0, 0, 0, -self.r_f * self.torque, 0],
                                [0, 0, 0, 0, 0, 0, 0, -self.r_f * self.torque],
                                [0, 0, 0, 0, 0, 0, 0, 0]]

            self.setup_keys_to_action()

    def apply_action(self, action):
        if isinstance(action, list):
            action = np.array(action)

        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('1'),): 0,
            (ord('2'),): 1,
            (ord('3'),): 2,
            (ord('4'),): 3,
            (ord('5'),): 4,
            (ord('6'),): 5,
            (ord('7'),): 6,
            (ord('8'),): 7,
            (ord('9'),): 8,
            (ord('0'),): 9,
            (ord('q'),): 10,
            (ord('w'),): 11,
            (ord('e'),): 12,
            (ord('r'),): 13,
            (ord('t'),): 14,
            (ord('y'),): 15,
            (): 4
        }


class Humanoid(WalkerBase):
    self_collision = True
    model_type = "MJCF"
    default_scale = 1
    glass_offset = 0.3

    def __init__(self, config):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "humanoid.xml", "torso", action_dim=17,
                            sensor_dim=40, power=0.41, scale=scale,
                            initial_pos=config['initial_pos'],
                            resolution=config["resolution"],
                            )
        self.glass_id = None
        self.is_discrete = config["is_discrete"]
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
            self.torque = 0.1
            self.action_list = np.concatenate((np.ones((1, 17)), np.zeros((1, 17)))).tolist()
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
            cid = p.createConstraint(humanoidId, -1, self.glass_id, -1, p.JOINT_FIXED,
                                     jointAxis=[0, 0, 0], parentFramePosition=[0, 0, self.glass_offset],
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

    def apply_action(self, a):

        if isinstance(action, list):
            action = np.array(action)

        if self.is_discrete:
            realaction = self.action_list[a]
        else:
            force_gain = 1
            for i, m, power in zip(range(17), self.motors, self.motor_power):
                m.set_motor_torque(float(force_gain * power * self.power * a[i]))
            # m.set_motor_torque(float(force_gain * power * self.power * np.clip(a[i], -1, +1)))

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,
            (): 1
        }


class Husky(WalkerBase):
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1

    def __init__(self, config):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale

        WalkerBase.__init__(self, "husky.urdf", "base_link", action_dim=4,
                            sensor_dim=17, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            resolution=config["resolution"],
                            )
        self.is_discrete = config["is_discrete"]

        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
            self.torque = 0.03
            self.action_list = [[self.torque, self.torque, self.torque, self.torque],
                                [-self.torque, -self.torque, -self.torque, -self.torque],
                                [self.torque, -self.torque, self.torque, -self.torque],
                                [-self.torque, self.torque, -self.torque, self.torque],
                                [0, 0, 0, 0]]

            self.setup_keys_to_action()
        else:
            action_high = 0.02 * np.ones([4])
            self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

    def apply_action(self, action):
        if isinstance(action, list):
            action = np.array(action)

        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

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
        top_xyz = self.parts["top_bumper_link"].pose().xyz()
        bottom_xyz = self.parts["base_link"].pose().xyz()
        alive = top_xyz[2] > bottom_xyz[2]
        return +1 if alive else -100  # 0.25 is central sphere rad, die if it scrapes the ground

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
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
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        self.is_discrete = config["is_discrete"]
        WalkerBase.__init__(self, "quadrotor.urdf", "base_link", action_dim=6,
                            sensor_dim=6, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            resolution=config["resolution"],
                            )
        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(7)

            self.action_list = [[1, 0, 0, 0, 0, 0],
                                [-1, 0, 0, 0, 0, 0],
                                [0, 1, 0, 0, 0, 0],
                                [0, -1, 0, 0, 0, 0],
                                [0, 0, 1, 0, 0, 0],
                                [0, 0, -1, 0, 0, 0],
                                [0, 0, 0, 0, 0, 0]
                                ]
            self.setup_keys_to_action()
        else:
            action_high = 0.02 * np.ones([6])
            self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)


    def apply_action(self, action):
        if isinstance(action, list):
            action = np.array(action)

        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action

        p.setGravity(0, 0, 0)
        p.resetBaseVelocity(self.robot_ids[0], realaction[:3], realaction[3:])

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## +x
            (ord('s'),): 1,  ## -x
            (ord('d'),): 2,  ## +y
            (ord('a'),): 3,  ## -y
            (ord('z'),): 4,  ## +z
            (ord('x'),): 5,  ## -z
            (): 6
        }


class Turtlebot(WalkerBase):
    mjcf_scaling = 1
    model_type = "URDF"
    default_scale = 1

    def __init__(self, config):
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "turtlebot/turtlebot.urdf", "base_link", action_dim=2,
                            sensor_dim=14, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            resolution=config["resolution"],
                            control='velocity',
                            )
        self.is_discrete = config["is_discrete"]

        self.velocity = config.get('velocity', 1.0)
        print('velocity', self.velocity)
        if self.is_discrete:
            # self.action_space = gym.spaces.Discrete(5)
            # self.action_list = [[self.velocity * 0.5, self.velocity * 0.5],
            #                     [-self.velocity * 0.5, -self.velocity * 0.5],
            #                     [self.velocity * 0.5, -self.velocity * 0.5],
            #                     [-self.velocity * 0.5, self.velocity * 0.5],
            #                     [0, 0]]
            self.action_space = gym.spaces.Discrete(3)
            self.action_list = [[self.velocity, self.velocity],
                                [self.velocity * 0.5, -self.velocity * 0.5],
                                [-self.velocity * 0.5, self.velocity * 0.5]]
            self.setup_keys_to_action()
        else:
            self.action_space = gym.spaces.Box(shape=(self.action_dim,), low=0.0, high=1.0, dtype=np.float32)
            self.action_low = -self.velocity * np.ones([self.action_dim])
            self.action_high = -self.action_low

    def apply_action(self, action):
        if isinstance(action, list):
            action = np.array(action)

        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
            (): 4
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
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "jr2_urdf/jr2.urdf", "base_link", action_dim=4,
                            sensor_dim=17, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            resolution=config["resolution"],
                            control=['velocity', 'velocity', 'position', 'position'],
                            )
        self.is_discrete = config["is_discrete"]
        self.velocity = config.get('velocity', 1.0)
        print('velocity', self.velocity)

        if self.is_discrete:
            self.action_space = gym.spaces.Discrete(5)
            self.action_list = [[self.velocity * 0.5, self.velocity * 0.5, 0, 0.2],
                                [-self.velocity * 0.5, -self.velocity * 0.5, 0, -0.2],
                                [self.velocity * 0.5, -self.velocity * 0.5, -0.5, 0],
                                [-self.velocity * 0.5, self.velocity * 0.5, 0.5, 0],
                                [0, 0, 0, 0]]

            self.setup_keys_to_action()
        else:

            action_high = self.velocity * np.ones([4])
            self.action_space = gym.spaces.Box(-action_high, action_high, dtype=np.float32)

    def apply_action(self, action):
        if isinstance(action, list):
            action = np.array(action)

        if self.is_discrete:
            realaction = self.action_list[action]
        else:
            realaction = action
        WalkerBase.apply_action(self, realaction)

    def setup_keys_to_action(self):
        self.keys_to_action = {
            (ord('w'),): 0,  ## forward
            (ord('s'),): 1,  ## backward
            (ord('d'),): 2,  ## turn right
            (ord('a'),): 3,  ## turn left
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
        self.config = config
        scale = config["robot_scale"] if "robot_scale" in config.keys() else self.default_scale
        WalkerBase.__init__(self, "jr2_urdf/jr2_kinova.urdf", "base_link", action_dim=12,
                            sensor_dim=33, power=2.5, scale=scale,
                            initial_pos=config['initial_pos'],
                            resolution=config["resolution"],
                            control=['velocity'] * 12,
                            )
        self.is_discrete = False
        self.wheel_velocity = config.get('wheel_velocity', 0.1)
        self.arm_velocity = config.get('arm_velocity', 0.01)

        # wheel_dim = 2
        # camera_dim = 2
        # arm_dim = 8
        # assert self.action_dim == wheel_dim + camera_dim + arm_dim
        self.action_low = np.array([-self.wheel_velocity] * 2 + [-self.arm_velocity] * 8)
        self.action_high = -self.action_low
        self.action_space = gym.spaces.Box(shape=(10,), low=-1.0, high=1.0, dtype=np.float32)
        # self.action_low = np.array([-self.wheel_velocity] * 2 + [-self.arm_velocity] * 1)
        # self.action_high = -self.action_low
        # self.action_space = gym.spaces.Box(shape=(3,), low=0.0, high=1.0)
        assert np.array_equal(self.action_low, -self.action_high)

        ##############################################################################################
        #
        # self.action_low = np.array([-self.velocity, -self.velocity,
        #                             -np.pi, 0.872664625997, 0.610865238198, -np.pi, -np.pi, -np.pi,
        #                             0.0, 0.0])
        # self.action_high = np.array([self.velocity, self.velocity,
        #                              np.pi, 5.41052068118, 5.67232006898, np.pi, np.pi, np.pi,
        #                              2.0, 2.0])
        # self.action_space = gym.spaces.Box(shape=(10,), low=0.0, high=1.0)
        # self.action_low = np.array([-self.velocity, -self.velocity, -np.pi])
        # self.action_high = np.array([self.velocity, self.velocity, np.pi])
        # self.action_space = gym.spaces.Box(shape=(3,), low=0.0, high=1.0)
        #self.action_low = np.array([-0.05, -0.05,
        #                            -np.pi, 0.872664625997, 0.610865238198, -np.pi])
        #self.action_high = np.array([0.05, 0.05,
        #                             np.pi, 5.41052068118, 5.67232006898, np.pi])
        #self.action_space = gym.spaces.Box(shape=(6,), low=0.0, high=1.0)
        # self.action_low = np.array([-0.05, -0.05,
        #                             -np.pi, 0.872664625997, 0.610865238198, -np.pi, -np.pi, -np.pi])
        # self.action_high = np.array([0.05, 0.05,
        #                              np.pi, 5.41052068118, 5.67232006898, np.pi, np.pi, np.pi])
        # self.action_space = gym.spaces.Box(shape=(8,), low=0.0, high=1.0)

    def apply_action(self, action):
        if isinstance(action, list):
            action = np.array(action)
        action = np.clip(action, self.action_space.low, self.action_space.high)
        normalized_action = self.action_high * action
        real_action = np.zeros(self.action_dim)
        real_action[:2] = normalized_action[:2]
        real_action[4:] = normalized_action[2:]
        # real_action[4] = normalized_action[2]
        #real_action[4:8] = normalized_action[2:]
        # real_action[4:10] = normalized_action[2:]
        WalkerBase.apply_action(self, real_action)

    def calc_state(self):
        base_state = WalkerBase.calc_state(self)
        angular_velocity = self.robot_body.angular_velocity()
        return np.concatenate((base_state, np.array(angular_velocity)))
