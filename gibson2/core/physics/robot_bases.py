# Author: pybullet, Zhiyang He

import pybullet as p
import gym, gym.spaces, gym.utils
import numpy as np
import os, inspect
import pybullet_data
from transforms3d.euler import euler2quat
from transforms3d import quaternions
from gibson2.utils.utils import quatFromXYZW, quatToXYZW
import gibson2

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)


class BaseRobot:
    """
    Base class for mujoco .xml/ROS urdf based agents.
    Handles object loading
    """

    def __init__(self, model_file, robot_name, scale=1):
        self.parts = None
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None

        self.robot_ids = None
        self.model_file = model_file
        self.robot_name = robot_name
        self.physics_model_dir = os.path.join(gibson2.assets_path, "models")
        self.scale = scale
        self.eyes = None
        print(self.model_file)
        if self.model_file[-4:] == 'urdf':
            self.model_type = 'URDF'
        else:
            self.model_type = 'MJCF'
        self.config = None
        self.np_random = None

    def load(self):
        ids = self._load_model()
        self.eyes = self.parts["eyes"]
        return ids

    def addToScene(self, bodies):
        assert len(bodies) == 1, 'robot body has length > 1'

        if self.parts is not None:
            parts = self.parts
        else:
            parts = {}

        if self.jdict is not None:
            joints = self.jdict
        else:
            joints = {}

        if self.ordered_joints is not None:
            ordered_joints = self.ordered_joints
        else:
            ordered_joints = []

        part_name, robot_name = p.getBodyInfo(bodies[0])
        part_name = part_name.decode("utf8")
        parts[part_name] = BodyPart(part_name,
                                    bodies,
                                    0,
                                    -1,
                                    self.scale,
                                    model_type=self.model_type)

        # By default, use base_link as self.robot_body
        if self.robot_name == part_name:
            self.robot_body = parts[part_name]

        for j in range(p.getNumJoints(bodies[0])):
            p.setJointMotorControl2(bodies[0],
                                    j,
                                    p.POSITION_CONTROL,
                                    positionGain=0.1,
                                    velocityGain=0.1,
                                    force=0)
            _, joint_name, joint_type, _, _, _, _, _, _, _, _, _, part_name, _, _, _, _ = p.getJointInfo(
                bodies[0], j)
            joint_name = joint_name.decode("utf8")
            part_name = part_name.decode("utf8")

            parts[part_name] = BodyPart(part_name,
                                        bodies,
                                        0,
                                        j,
                                        self.scale,
                                        model_type=self.model_type)

            # If self.robot_name is not base_link, but a body part, use it as self.robot_body
            if self.robot_name == part_name:
                self.robot_body = parts[part_name]

            if joint_name[:6] == "ignore":
                Joint(joint_name, bodies, 0, j, self.scale,
                      model_type=self.model_type).disable_motor()
                continue

            if joint_name[:8] != "jointfix" and joint_type != p.JOINT_FIXED:
                joints[joint_name] = Joint(joint_name,
                                           bodies,
                                           0,
                                           j,
                                           self.scale,
                                           model_type=self.model_type)
                ordered_joints.append(joints[joint_name])
                joints[joint_name].power_coef = 100.0

        if self.robot_body is None:
            raise Exception('robot body not initialized.')

        return parts, joints, ordered_joints, self.robot_body

    def _load_model(self):
        if self.model_type == "MJCF":
            self.robot_ids = p.loadMJCF(os.path.join(self.physics_model_dir, self.model_file),
                                        flags=p.URDF_USE_SELF_COLLISION +
                                        p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
        if self.model_type == "URDF":
            self.robot_ids = (p.loadURDF(os.path.join(self.physics_model_dir, self.model_file),
                                         globalScaling=self.scale), )

        self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(
            self.robot_ids)
        return self.robot_ids

    def robot_specific_reset(self):
        raise NotImplementedError

    def calc_state(self):
        raise NotImplementedError

    def reset(self):
        if self.robot_ids is None:
            self._load_model()

        self.robot_body.reset_orientation(
            quatToXYZW(euler2quat(*self.config["initial_orn"]), 'wxyz'))
        self.robot_body.reset_position(self.config["initial_pos"])
        self.reset_random_pos()
        self.robot_specific_reset()

        state = self.calc_state()
        return state

    def reset_random_pos(self):
        '''Add randomness to resetted initial position
        '''
        if not self.config["random"]["random_initial_pose"]:
            return

        pos = self.robot_body.get_position()
        orn = self.robot_body.get_orientation()

        x_range = self.config["random"]["random_init_x_range"]
        y_range = self.config["random"]["random_init_y_range"]
        z_range = self.config["random"]["random_init_z_range"]
        r_range = self.config["random"]["random_init_rot_range"]

        new_pos = [
            pos[0] + self.np_random.uniform(low=x_range[0], high=x_range[1]),
            pos[1] + self.np_random.uniform(low=y_range[0], high=y_range[1]),
            pos[2] + self.np_random.uniform(low=z_range[0], high=z_range[1])
        ]
        new_orn = quaternions.qmult(
            quaternions.axangle2quat([1, 0, 0],
                                     self.np_random.uniform(low=r_range[0], high=r_range[1])), orn)

        self.robot_body.reset_orientation(new_orn)
        self.robot_body.reset_position(new_pos)

    def reset_new_pose(self, pos, orn):
        self.robot_body.reset_orientation(orn)
        self.robot_body.reset_position(pos)

    def calc_potential(self):
        return 0


class Pose_Helper:
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.get_position()

    def rpy(self):
        return p.getEulerFromQuaternion(self.body_part.get_orientation())

    def orientation(self):
        return self.body_part.get_orientation()


class BodyPart:
    def __init__(self, body_name, bodies, body_index, body_part_index, scale, model_type):
        self.bodies = bodies
        self.body_name = body_name
        self.body_index = body_index
        self.body_part_index = body_part_index
        if model_type == "MJCF":
            self.scale = scale
        else:
            self.scale = 1
        self.initialPosition = self.get_position() / self.scale
        self.initialOrientation = self.get_orientation()
        self.bp_pose = Pose_Helper(self)

    def get_name(self):
        return self.body_name

    def _state_fields_of_pose_of(self, body_id, link_id=-1):
        """Calls native pybullet method for getting real (scaled) robot body pose

           Note that there is difference between xyz in real world scale and xyz
           in simulation. Thus you should never call pybullet methods directly
        """
        if link_id == -1:
            (x, y, z), (a, b, c, d) = p.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = p.getLinkState(body_id, link_id)
        x, y, z = x * self.scale, y * self.scale, z * self.scale
        return np.array([x, y, z, a, b, c, d])

    def _set_fields_of_pose_of(self, pos, orn):
        """Calls native pybullet method for setting real (scaled) robot body pose"""
        p.resetBasePositionAndOrientation(self.bodies[self.body_index],
                                          np.array(pos) / self.scale, orn)

    def get_pose(self):
        return self._state_fields_of_pose_of(self.bodies[self.body_index], self.body_part_index)

    def get_position(self):
        """Get position of body part
           Position is defined in real world scale """
        return self.get_pose()[:3]

    def get_orientation(self):
        """Get orientation of body part
           Orientation is by default defined in [x,y,z,w]"""
        return self.get_pose()[3:]

    def get_rpy(self):
        """Get roll, pitch and yaw of body part
           [roll, pitch, yaw]"""
        return p.getEulerFromQuaternion(self.get_orientation())

    def set_position(self, position):
        """Get position of body part
           Position is defined in real world scale """
        self._set_fields_of_pose_of(position, self.get_orientation())

    def set_orientation(self, orientation):
        """Get position of body part
           Orientation is defined in [x,y,z,w]"""
        self._set_fields_of_pose_of(self.current_position(), orientation)

    def set_pose(self, position, orientation):
        self._set_fields_of_pose_of(position, orientation)

    def current_position(self):    # Synonym method
        return self.get_position()

    def current_orientation(self):    # Synonym method
        return self.get_orientation()

    def reset_position(self, position):    # Backward compatibility
        self.set_position(position)

    def reset_orientation(self, orientation):    # Backward compatibility
        self.set_orientation(orientation)

    def reset_pose(self, position, orientation):    # Backward compatibility
        self.set_pose(position, orientation)

    def velocity(self):
        if self.body_part_index == -1:
            (vx, vy, vz), _ = p.getBaseVelocity(self.bodies[self.body_index])
        else:
            (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vyaw) = p.getLinkState(
                self.bodies[self.body_index], self.body_part_index, computeLinkVelocity=1)
        return np.array([vx, vy, vz])

    def angular_velocity(self):
        if self.body_part_index == -1:
            _, (vr, vp, vyaw) = p.getBaseVelocity(self.bodies[self.body_index])
        else:
            (x, y, z), (a, b, c, d), _, _, _, _, (vx, vy, vz), (vr, vp, vyaw) = p.getLinkState(
                self.bodies[self.body_index], self.body_part_index, computeLinkVelocity=1)
        return np.array([vr, vp, vyaw])

    def contact_list(self):
        return p.getContactPoints(self.bodies[self.body_index], -1, self.body_part_index, -1)


class Joint:
    def __init__(self, joint_name, bodies, body_index, joint_index, scale, model_type):
        self.bodies = bodies
        self.body_index = body_index
        self.joint_index = joint_index
        self.joint_name = joint_name
        _, _, self.joint_type, _, _, _, _, _, self.lower_limit, self.upper_limit, _, self.max_velocity, _, _, _, _, _ \
            = p.getJointInfo(self.bodies[self.body_index], self.joint_index)
        self.power_coeff = 0
        if model_type == "MJCF":
            self.scale = scale
        else:
            self.scale = 1
        if self.joint_type == p.JOINT_PRISMATIC:
            self.upper_limit *= self.scale
            self.lower_limit *= self.scale

    def __str__(self):
        return "idx: {}, name: {}".format(self.joint_index, self.joint_name)

    def get_state(self):
        """Get state of joint
           Position is defined in real world scale """
        x, vx, _, trq = p.getJointState(self.bodies[self.body_index], self.joint_index)
        if self.joint_type == p.JOINT_PRISMATIC:
            x *= self.scale
            vx *= self.scale
        return x, vx, trq

    def set_state(self, x, vx):
        """Set state of joint
           x is defined in real world scale """
        if self.joint_type == p.JOINT_PRISMATIC:
            x /= self.scale
            vx /= self.scale
        p.resetJointState(self.bodies[self.body_index], self.joint_index, x, vx)

    def get_relative_state(self):
        pos, vel, trq = self.get_state()

        # normalize position to [-1, 1]
        if self.lower_limit < self.upper_limit:
            pos = 2 * (pos - 0.5 * (self.lower_limit + self.upper_limit)) / (self.upper_limit -
                                                                             self.lower_limit)

        # (try to) normalize velocity to [-1, 1]
        if self.max_velocity > 0:
            vel /= self.max_velocity
        elif self.joint_type == p.JOINT_REVOLUTE:
            vel *= 0.1
        else:
            vel *= 0.5
        return pos, vel, trq

    def set_position(self, position):
        """Set position of joint
           Position is defined in real world scale """
        if self.joint_type == p.JOINT_PRISMATIC:
            position = np.array(position) / self.scale
        p.setJointMotorControl2(self.bodies[self.body_index],
                                self.joint_index,
                                p.POSITION_CONTROL,
                                targetPosition=position)

    def set_velocity(self, velocity):
        """Set velocity of joint
           Velocity is defined in real world scale """
        if self.joint_type == p.JOINT_PRISMATIC:
            velocity = np.array(velocity) / self.scale
        p.setJointMotorControl2(self.bodies[self.body_index],
                                self.joint_index,
                                p.VELOCITY_CONTROL,
                                targetVelocity=velocity)    # , positionGain=0.1, velocityGain=0.1)

    def set_torque(self, torque):
        p.setJointMotorControl2(bodyIndex=self.bodies[self.body_index],
                                jointIndex=self.joint_index,
                                controlMode=p.TORQUE_CONTROL,
                                force=torque)    # , positionGain=0.1, velocityGain=0.1)

    def reset_state(self, pos, vel):
        self.set_state(pos, vel)

    def disable_motor(self):
        p.setJointMotorControl2(self.bodies[self.body_index],
                                self.joint_index,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=0,
                                targetVelocity=0,
                                positionGain=0.1,
                                velocityGain=0.1,
                                force=0)

    def get_joint_relative_state(self):    # Synonym method
        return self.get_relative_state()

    def set_motor_position(self, pos):    # Synonym method
        return self.set_position(pos)

    def set_motor_torque(self, torque):    # Synonym method
        return self.set_torque(torque)

    def set_motor_velocity(self, vel):    # Synonym method
        return self.set_velocity(vel)

    def reset_joint_state(self, position, velocity):    # Synonym method
        return self.reset_state(position, velocity)

    def current_position(self):    # Backward compatibility
        return self.get_state()

    def current_relative_position(self):    # Backward compatibility
        return self.get_relative_state()

    def reset_current_position(self, position, velocity):    # Backward compatibility
        self.reset_state(position, velocity)

    def reset_position(self, position, velocity):    # Backward compatibility
        self.reset_state(position, velocity)
