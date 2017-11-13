## Author: pybullet, Zhiyang He

import pybullet as p
import gym, gym.spaces, gym.utils
import numpy as np
import os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0,parentdir)
import pybullet_data


class BaseRobot:
    """
    Base class for mujoco .xml/ROS urdf based agents.
    Handles object loading
    """

    def __init__(self, model_file, robot_name, scale = 1):
        self.parts = None
        self.jdict = None
        self.ordered_joints = None
        self.robot_body = None

        self.robot_ids = None
        self.model_file = model_file
        self.robot_name = robot_name
        self.physics_model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.scale = scale

    def addToScene(self, bodies):
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

        dump = 0
        for i in range(len(bodies)):
            if p.getNumJoints(bodies[i]) == 0:
                part_name, robot_name = p.getBodyInfo(bodies[i], 0)
                robot_name = robot_name.decode("utf8")
                part_name = part_name.decode("utf8")
                parts[part_name] = BodyPart(part_name, bodies, i, -1, self.scale, model_type=self.model_type)
            for j in range(p.getNumJoints(bodies[i])):
                p.setJointMotorControl2(bodies[i],j,p.POSITION_CONTROL,positionGain=0.1,velocityGain=0.1,force=0)
                _,joint_name,joint_type,_,_,_,_,_,_,_,_,_,part_name = p.getJointInfo(bodies[i], j)

                joint_name = joint_name.decode("utf8")
                part_name = part_name.decode("utf8")

                if dump: print("ROBOT PART '%s'" % part_name)
                if dump: print("ROBOT JOINT '%s'" % joint_name) # limits = %+0.2f..%+0.2f effort=%0.3f speed=%0.3f" % ((joint_name,) + j.limits()) )
                parts[part_name] = BodyPart(part_name, bodies, i, j, self.scale, model_type=self.model_type)

                if part_name == self.robot_name:
                    self.robot_body = parts[part_name]

                if i == 0 and j == 0 and self.robot_body is None:  # if nothing else works, we take this as robot_body
                    parts[self.robot_name] = BodyPart(self.robot_name, bodies, 0, -1, self.scale, model_type=self.model_type)
                    self.robot_body = parts[self.robot_name]

                #print(joint_name)
                if joint_name[:6] == "ignore":
                    Joint(joint_name, bodies, i, j, self.scale).disable_motor()
                    continue

                if joint_name[:8] != "jointfix" and joint_type != p.JOINT_FIXED:
                    joints[joint_name] = Joint(joint_name, bodies, i, j, self.scale, model_type=self.model_type)
                    ordered_joints.append(joints[joint_name])

                    joints[joint_name].power_coef = 100.0

        debugmode = 0
        if debugmode:
            for j in ordered_joints:
                print(j, j.power_coef)
        return parts, joints, ordered_joints, self.robot_body

    def reset(self):
        #if self.parts:
        #    [p.removeBody(self.parts[p_name].bodyIndex) for p_name in self.parts]
        #print(os.path.join(os.path.dirname(os.path.abspath(__file__)),"models", self.model_file))
        ## Use self-collision

        if self.robot_ids is None:
            if self.model_type == "MJCF":
                self.robot_ids = p.loadMJCF(os.path.join(self.physics_model_dir, self.model_file), flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS)
            if self.model_type == "URDF":
                self.robot_ids = (p.loadURDF(os.path.join(self.physics_model_dir, self.model_file), flags=p.URDF_USE_SELF_COLLISION+p.URDF_USE_SELF_COLLISION_EXCLUDE_ALL_PARENTS, globalScaling = self.scale), )
            self.parts, self.jdict, self.ordered_joints, self.robot_body = self.addToScene(self.robot_ids)
            #print(self.ordered_joints)
        #print("body before", self.robot_body)
    
        #print("body after", self.robot_body)
        self.robot_specific_reset()

        ## reset returns robot sensor state
        state = self.calc_state()  # optimization: calc_state() can calculate something in self.* for calc_potential() to use
        self.eyes = self.parts["eyes"]
        return state


    def calc_potential(self):
        return 0

class Pose_Helper: # dummy class to comply to original interface
    def __init__(self, body_part):
        self.body_part = body_part

    def xyz(self):
        return self.body_part.current_position()

    def rpy(self):
        return p.getEulerFromQuaternion(self.body_part.current_orientation())

    def orientation(self):
        return self.body_part.current_orientation()

class BodyPart:
    def __init__(self, body_name, bodies, bodyIndex, bodyPartIndex, scale, model_type):
        self.bodies = bodies
        self.body_name = body_name
        self.bodyIndex = bodyIndex
        self.bodyPartIndex = bodyPartIndex
        if model_type=="MJCF":
            self.scale = scale
        else:
            self.scale = 1
        self.initialPosition = self.current_position() / self.scale
        self.initialOrientation = self.current_orientation()
        self.bp_pose = Pose_Helper(self)

    def get_name(self):
        return self.body_name

    def state_fields_of_pose_of(self, body_id, link_id=-1):  # a method you will most probably need a lot to get pose and orientation
        if link_id == -1:
            (x, y, z), (a, b, c, d) = p.getBasePositionAndOrientation(body_id)
        else:
            (x, y, z), (a, b, c, d), _, _, _, _ = p.getLinkState(body_id, link_id)
        x, y, z = x * self.scale, y * self.scale, z * self.scale
        return np.array([x, y, z, a, b, c, d])

    def get_pose(self):
        return self.state_fields_of_pose_of(self.bodies[self.bodyIndex], self.bodyPartIndex)

    def speed(self):
        if self.bodyPartIndex == -1:
            (vx, vy, vz), _ = p.getBaseVelocity(self.bodies[self.bodyIndex])
        else:
            (x,y,z), (a,b,c,d), _,_,_,_, (vx, vy, vz), (vr,vp,vy) = p.getLinkState(self.bodies[self.bodyIndex], self.bodyPartIndex, computeLinkVelocity=1)
        return np.array([vx, vy, vz])

    def current_position(self):
        return self.get_pose()[:3]

    def current_orientation(self):
        return self.get_pose()[3:]

    def reset_position(self, position):
        print("robot scale", self.scale)
        p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], np.array(position) / self.scale, self.current_orientation())

    def reset_orientation(self, orientation):
        p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], self.current_position() / self.scale, orientation)

    def reset_pose(self, position, orientation):
        p.resetBasePositionAndOrientation(self.bodies[self.bodyIndex], np.array(position) / self.scale, orientation)

    def pose(self):
        return self.bp_pose

    def contact_list(self):
        return p.getContactPoints(self.bodies[self.bodyIndex], -1, self.bodyPartIndex, -1)


class Joint:
    def __init__(self, joint_name, bodies, bodyIndex, jointIndex, scale, model_type):
        self.bodies = bodies
        self.bodyIndex = bodyIndex
        self.jointIndex = jointIndex
        self.joint_name = joint_name
        _,_,_,_,_,_,_,_,self.lowerLimit, self.upperLimit,_,_,_ = p.getJointInfo(self.bodies[self.bodyIndex], self.jointIndex)
        self.power_coeff = 0
        if model_type=="mjcf":
            self.scale = scale
        else:
            self.scale = 1

    def set_state(self, x, vx):
        p.resetJointState(self.bodies[self.bodyIndex], self.jointIndex, x, vx)

    def current_position(self): # just some synonyme method
        state = self.get_state
        state[:3] = state[:3] * self.scale
        return state

    def current_relative_position(self):
        pos, vel = self.get_state()
        pos_mid = 0.5 * (self.lowerLimit + self.upperLimit);
        return (
            2 * (pos - pos_mid) / (self.upperLimit - self.lowerLimit),
            0.1 * vel
        )

    def get_state(self):
        x, vx,_,_ = p.getJointState(self.bodies[self.bodyIndex],self.jointIndex)
        return x * self.scale, vx

    def set_position(self, position):
        p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,p.POSITION_CONTROL, targetPosition=np.array(position) / self.scale)

    def set_velocity(self, velocity):
        p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,p.VELOCITY_CONTROL, targetVelocity=velocity)

    def set_motor_torque(self, torque): # just some synonyme method
        self.set_torque(torque)

    def set_torque(self, torque):
        p.setJointMotorControl2(bodyIndex=self.bodies[self.bodyIndex], jointIndex=self.jointIndex, controlMode=p.TORQUE_CONTROL, force=torque) #, positionGain=0.1, velocityGain=0.1)

    def reset_current_position(self, position, velocity): # just some synonyme method
        self.reset_position(position / self.scale, velocity)

    def reset_position(self, position, velocity):
        p.resetJointState(self.bodies[self.bodyIndex],self.jointIndex,targetValue= np.array(position) / self.scale, targetVelocity=velocity)
        self.disable_motor()

    def disable_motor(self):
        p.setJointMotorControl2(self.bodies[self.bodyIndex],self.jointIndex,controlMode=p.POSITION_CONTROL, targetPosition=0, targetVelocity=0, positionGain=0.1, velocityGain=0.1, force=0)
