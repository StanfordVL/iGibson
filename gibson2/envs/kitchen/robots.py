import numpy as np
from contextlib import contextmanager
from copy import deepcopy

import pybullet as p
import gibson2.external.pybullet_tools.utils as PBU
import gibson2.external.pybullet_tools.transformations as T
import gibson2.envs.kitchen.plan_utils as PU
from gibson2.envs.kitchen.env_utils import set_collision_between


GRIPPER_OPEN = -1
GRIPPER_CLOSE = 1


class Gripper(object):
    def __init__(
            self,
            joint_names=(),
            finger_link_names=(),
            joint_min=(0.00, 0.00),
            joint_max=(1., 1.),
            use_magic_grasp=True,
            env=None
    ):
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.joint_names = joint_names
        self.finger_link_names = finger_link_names
        self.grasp_cid = None
        self.use_magic_grasp = use_magic_grasp
        self.magic_ungrasp_delay = 0
        self.magic_ungrasp_delay_counter = 0

        self.body_id = None
        self.filename = None
        self.env = env

    def load(self, filename=None, body_id=None, scale=1.):
        self.filename = filename
        if self.filename is not None:
            self.body_id = p.loadURDF(
                self.filename, globalScaling=scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        else:
            self.body_id = body_id
        # change friction to simulate rubber
        for l in self.finger_links:
            p.changeDynamics(self.body_id, l, lateralFriction=10.)

        # set to open
        for i, jointIndex in enumerate(self.joints):
            p.resetJointState(self.body_id, jointIndex, self.joint_max[i])

    @property
    def joints(self):
        return PBU.joints_from_names(self.body_id, self.joint_names)

    @property
    def finger_links(self):
        return [PBU.link_from_name(self.body_id, n) for n in self.finger_link_names]

    def get_grasped_id(self, candidate_ids):
        """Try to figure out which object the gripper is grasping"""
        for oid in candidate_ids:
            for l in PBU.get_all_links(oid):
                fn_contact = []
                for fn in self.finger_link_names:
                    # if PBU.pairwise_link_collision(self.body_id, PBU.link_from_name(self.body_id, fn), oid, l):
                    if len(p.getContactPoints(self.body_id, oid, PBU.link_from_name(self.body_id, fn), l)) > 0:
                        fn_contact.append(fn)
                num_left = len([fn for fn in fn_contact if fn.startswith("left")])
                num_right = len([fn for fn in fn_contact if fn.startswith("right")])
                if num_left > 0 and num_right > 0:
                    return oid, l
        else:
            return None

    def get_joint_positions(self):
        return np.array([p.getJointState(self.body_id, j)[0] for j in self.joints])

    def grasp(self):
        self.set_joint_positions(self.joint_min, force=100.)
        if self.use_magic_grasp:
            self._magic_grasp(joint_type=p.JOINT_FIXED)

    def ungrasp(self):
        self.set_joint_positions(self.joint_max, force=100.)
        self.reset_joint_positions(self.joint_max)
        if self.use_magic_grasp:
            self._magic_ungrasp()

    def set_joint_forces(self, forces):
        assert len(forces) == len(self.joints)
        for i, joint_index in enumerate(self.joints):
            p.setJointMotorControl2(self.body_id, joint_index, p.TORQUE_CONTROL, force=forces[i])

    def set_joint_positions(self, positions, force=100.):
        assert len(positions) == len(self.joints)
        for i, joint_idx in enumerate(self.joints):
            pos = positions[i]
            pos = max(self.joint_min[i], pos)
            pos = min(self.joint_max[i], pos)
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=pos, force=force)

    def reset_joint_positions(self, positions):
        assert len(positions) == len(self.joints)
        for i, j in enumerate(self.joints):
            p.resetJointState(self.body_id, j, targetValue=positions[i])

    def _magic_grasp(self, joint_type=p.JOINT_FIXED):
        """Perform magic grasp by creating a constraint between the gripper and object."""
        if self.grasp_cid is not None:
            return

        # check if the gripper has grasped anything
        ret = self.get_grasped_id(self.env.objects.body_ids)
        if ret is None:
            return

        target_id, target_link = ret
        # return

        obj_pos, obj_orn = p.getBasePositionAndOrientation(target_id)
        gripper_pos, gripper_orn = p.getBasePositionAndOrientation(self.body_id)
        grasp_pose = p.multiplyTransforms(*p.invertTransform(gripper_pos, gripper_orn), obj_pos, obj_orn)

        self.grasp_cid = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=-1,
            childBodyUniqueId=target_id,
            childLinkIndex=target_link,
            jointType=joint_type,
            jointAxis=(0, 0, 0),
            parentFramePosition=grasp_pose[0],
            childFramePosition=[0, 0, 0],
            parentFrameOrientation=grasp_pose[1],
            childFrameOrientation=(0, 0, 0, 1),
        )

        # print("grasped {}, link {}".format(self.env.objects.body_id_to_name(target_id), target_link))

    def _magic_ungrasp(self):
        if self.grasp_cid is not None and self.magic_ungrasp_delay_counter >= self.magic_ungrasp_delay:
            p.removeConstraint(self.grasp_cid)
            # print("ungrasp")
            self.grasp_cid = None
            self.magic_ungrasp_delay_counter = 0
        else:
            self.magic_ungrasp_delay_counter += 1


class Arm(object):
    def __init__(self, joint_names):
        self.body_id = None
        self.filename = None
        self.joint_names = joint_names

    @property
    def joints(self):
        return PBU.joints_from_names(self.body_id, self.joint_names)

    def load(self, filename=None, body_id=None, scale=1.):
        self.filename = filename
        if self.filename is not None:
            self.body_id = p.loadURDF(self.filename, globalScaling=scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        else:
            self.body_id = body_id

    def set_joint_positions(self, positions, force=100.):
        assert len(positions) == len(self.joints)
        for i, joint_idx in enumerate(self.joints):
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=positions[i], force=force)

    def reset_joint_positions(self, positions):
        assert len(positions) == len(self.joints)
        for i, j in enumerate(self.joints):
            p.resetJointState(self.body_id, j, targetValue=positions[i])


class Robot(object):
    def __init__(self, eef_link_name, init_base_pose, gripper=None, arm=None):
        super(Robot, self).__init__()
        self.eef_link_name = eef_link_name
        self.gripper = gripper
        self.arm = arm

        if self.arm is not None and self.gripper is not None:
            assert self.arm.body_id == self.gripper.body_id
        if self.arm is not None:
            assert isinstance(arm, Arm)
            self.body_id = arm.body_id
        elif self.gripper is not None:
            assert isinstance(gripper, Gripper)
            self.body_id = gripper.body_id
        else:
            raise ValueError("at least one of arm and gripper should be provided")

        p.resetBasePositionAndOrientation(self.body_id, init_base_pose[0], init_base_pose[1])

    def reset(self):
        pass

    @property
    def eef_link_index(self):
        return PBU.link_from_name(self.body_id, self.eef_link_name)

    def reset_base_position_orientation(self, pos, orn):
        return p.resetBasePositionAndOrientation(self.body_id, pos, orn)

    def get_eef_position_orientation(self):
        return p.getLinkState(self.body_id, self.eef_link_index)[:2]

    def get_eef_velocity(self):
        return p.getLinkState(self.body_id, self.eef_link_index)[-2:]  # linear, angular vel

    def get_eef_position(self):
        return self.get_eef_position_orientation()[0]

    def get_eef_orientation(self):
        return self.get_eef_position_orientation()[1]

    def set_eef_position_orientation(self, pos, orn):
        raise NotImplementedError

    def reset_eef_position_orientation(self, pos, orn):
        raise NotImplementedError

    def set_relative_eef_position_orientation(self, delta_pos, delta_orn):
        cpos, corn = self.get_eef_position_orientation()
        new_pos = tuple([cp + p for cp, p in zip(cpos, delta_pos)])
        new_orn = T.quaternion_multiply(delta_orn, corn)
        self.set_eef_position_orientation(new_pos, new_orn)

    def set_eef_position(self, pos):
        self.set_eef_position_orientation(pos, self.get_eef_orientation())

    def reset_eef_position(self, pos):
        self.reset_eef_position_orientation(pos, self.get_eef_orientation())


class PlannableRobot(Robot):
    def __init__(self, eef_link_name, init_base_pose, arm, gripper=None, plannable_joint_names=None):
        super(PlannableRobot, self).__init__(
            eef_link_name=eef_link_name,
            init_base_pose=init_base_pose,
            gripper=gripper,
            arm=arm
        )

        assert arm is not None
        self.plannable_joint_names = plannable_joint_names
        if plannable_joint_names is None:
            self.plannable_joint_names = arm.joint_names

    @property
    def plannable_joints(self):
        return [PBU.joint_from_name(self.body_id, jn) for jn in self.plannable_joint_names]

    def get_plannable_joint_positions(self):
        return [p.getJointState(self.body_id, joint_idx)[0] for joint_idx in self.plannable_joints]

    def reset_plannable_joint_positions(self, conf):
        assert len(conf) == len(self.plannable_joints)
        for i, joint_idx in enumerate(self.plannable_joints):
            p.resetJointState(self.body_id, joint_idx, targetValue=conf[i])

    def set_plannable_joint_positions(self, conf, force=100.):
        assert len(conf) == len(self.plannable_joints)
        for i, joint_idx in enumerate(self.plannable_joints):
            p.setJointMotorControl2(self.body_id, joint_idx, p.POSITION_CONTROL, targetPosition=conf[i])
        print(conf, self.get_plannable_joint_positions())

    def reset_eef_position_orientation(self, pos, orn):
        conf = PU.inverse_kinematics(self.body_id, self.eef_link_index, self.plannable_joints, target_pose=(pos, orn))
        self.reset_plannable_joint_positions(conf)

    def set_eef_position_orientation(self, pos, orn, force=100.):
        conf = PU.inverse_kinematics(self.body_id, self.eef_link_index, self.plannable_joints, target_pose=(pos, orn))
        self.set_plannable_joint_positions(conf)

    def plan_joint_path(self, target_pose, obstacles, resolutions=None):
        return PU.plan_joint_path(
            self.body_id,
            self.eef_link_name,
            self.plannable_joint_names,
            target_pose,
            obstacles=obstacles,
            resolutions=resolutions
        )

    def inverse_kinematics(self, target_pose):
        return PU.inverse_kinematics(self.body_id, self.eef_link_index, self.plannable_joints, target_pose=target_pose)


class PlannerRobot(PlannableRobot):
    def __init__(self, eef_link_name, init_base_pose, arm, gripper=None, plannable_joint_names=None, plan_objects=None):
        super(PlannerRobot, self).__init__(
            eef_link_name=eef_link_name,
            init_base_pose=init_base_pose,
            arm=arm,
            gripper=gripper,
            plannable_joint_names=plannable_joint_names
        )
        self.ref_robot = None
        self.plan_objects = plan_objects

    def setup(self, robot, hide_planner=True):
        self.ref_robot = robot
        self.disable_collision_with(PBU.get_bodies())
        self.synchronize(robot)
        if hide_planner:
            for l in PBU.get_all_links(self.body_id):
                p.changeVisualShape(self.body_id, l, rgbaColor=(0, 0, 1, 0))

    def disable_collision_with(self, others):
        for other in others:
            set_collision_between(self.body_id, other, 0)

    def enable_collision_with(self, others):
        for other in others:
            set_collision_between(self.body_id, other, 1)

    @contextmanager
    def collision_enabled_with(self, others):
        self.enable_collision_with(others)
        yield
        self.disable_collision_with(others)

    def synchronize(self, robot=None):
        if robot is None:
            robot = self.ref_robot
        assert isinstance(robot, Robot)
        # synchronize base pose if the shadow gripper has the same model as the actuated gripper
        # base_pos, base_orn = p.getBasePositionAndOrientation(robot.body_id)
        # p.resetBasePositionAndOrientation(self.body_id, base_pos, base_orn)

        # otherwise, synchronize by computing inverse kinematics wrt to eef link
        g_pose = robot.get_eef_position_orientation()
        conf = self.inverse_kinematics(g_pose)
        self.reset_plannable_joint_positions(conf)

        if len(robot.gripper.joints) == len(self.gripper.joints):
            self.gripper.reset_joint_positions(robot.gripper.get_joint_positions())
        if self.plan_objects is not None:
            self.plan_objects.synchronize()

    def plan_joint_path(self, target_pose, obstacles, resolutions=None, attachment_ids=(), synchronize=True):
        if synchronize:
            self.synchronize()  # synchronize planner with the robot

        if self.plan_objects is not None:
            attachment_ids = tuple([self.plan_objects.get_visual_copy_of(bid).body_id for bid in attachment_ids])

        # don't count attached objects as obstacles when doing motion planning
        obstacles = tuple([o for o in obstacles if o not in attachment_ids])

        with self.collision_enabled_with(obstacles):
            path = PU.plan_joint_path(
                self.body_id,
                self.eef_link_name,
                self.plannable_joint_names,
                target_pose,
                obstacles=obstacles,
                resolutions=resolutions,
                attachment_ids=attachment_ids
            )
        return path


class ConstraintActuatedRobot(Robot):
    def __init__(self, eef_link_name, init_base_pose, gripper):
        super(ConstraintActuatedRobot, self).__init__(
            eef_link_name=eef_link_name,
            init_base_pose=init_base_pose,
            gripper=gripper
        )
        self._create_eef_constraint()

    def _create_eef_constraint(self):
        gripper_base_pose = p.getBasePositionAndOrientation(self.body_id)
        self.cid = p.createConstraint(
            parentBodyUniqueId=self.body_id,
            parentLinkIndex=self.eef_link_index,
            childBodyUniqueId=-1,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=(0, 0, 0),
            parentFramePosition=(0, 0, 0),
            childFramePosition=gripper_base_pose[0],  # gripper base position
        )
        p.changeConstraint(self.cid, maxForce=10000)
        self.set_eef_position_orientation(*self.get_eef_position_orientation())

    def reset_base_position_orientation(self, pos, orn):
        """Remove constraint before resetting the base to avoid messing up the simulation"""
        p.removeConstraint(self.cid)
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)
        self._create_eef_constraint()

    def set_eef_position_orientation(self, pos, orn):
        p.changeConstraint(self.cid, pos, orn, maxForce=10000)
        # for _ in range(20):
        #     if np.allclose(pos, self.get_eef_position(), atol=1e-3):
        #         break
        #     # p.changeConstraint(self.cid, pos, orn)
        #     p.stepSimulation()
        # else:
        #     print("ooe")


class ConstraintTargetActuatedRobot(ConstraintActuatedRobot):
    def __init__(self, eef_link_name, init_base_pose, gripper):
        super(ConstraintTargetActuatedRobot, self).__init__(
            eef_link_name=eef_link_name,
            init_base_pose=init_base_pose,
            gripper=gripper
        )
        self._target = None

    def reset(self):
        self._target = self.get_eef_position_orientation()

    @property
    def target(self):
        return deepcopy(self._target)

    def set_relative_eef_position_orientation(self, delta_pos, delta_orn):
        new_pos = np.array(self._target[0]) + np.array(delta_pos)
        new_orn = T.quaternion_multiply(delta_orn, self._target[1])
        self._target = (tuple(new_pos.tolist()), tuple(new_orn.tolist()))
        self.set_eef_position_orientation(*self._target)


class JointActuatedRobot(Robot):
    def __init__(self, eef_link_name, init_base_pose, gripper, arm):
        super(JointActuatedRobot, self).__init__(
            eef_link_name=eef_link_name,
            init_base_pose=init_base_pose,
            gripper=gripper,
            arm=arm
        )

    @property
    def actuated_joints(self):
        return self.arm.joints

    def set_actuated_joint_positions(self, positions):
        return self.arm.set_joint_positions(positions=positions, force=100000)

    def close_loop_joint_control(self, target, tolerance=1e-3):
        assert (len(self.actuated_joints) == len(target))
        positions = PBU.get_joint_positions(self.body_id, self.actuated_joints)
        for _ in range(20):
            if np.allclose(target, positions, atol=tolerance, rtol=0):
                break
            PBU.control_joints(self.body_id, self.actuated_joints, target)
            p.stepSimulation()
            positions = PBU.get_joint_positions(self.body_id, self.actuated_joints)
            print(target, positions)