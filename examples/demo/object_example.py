from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, Object
import gibson2
import os
import pybullet as p
import pybullet_data
import time
import numpy as np
import gibson2.external.pybullet_tools.transformations as T
import gibson2.external.pybullet_tools.utils as PBU
from contextlib import contextmanager
from copy import deepcopy


"""
Task plans -> skill parameters
Parameterized skill library
Skills + parameters -> joint-space motion plan
Motion plan -> task-space path
task-space path -> gripper actuation
"""
GRIPPER_OPEN = 0
GRIPPER_CLOSE = 1


class Path(object):
    def __init__(self, arm_path=None, gripper_path=None):
        self._arm_path = arm_path
        self._gripper_path = gripper_path

    @property
    def arm_path(self):
        return self._arm_path

    @property
    def gripper_path(self):
        if self._gripper_path is not None:
            assert len(self._gripper_path) == len(self._arm_path)
        return self._gripper_path

    def append(self, arm_state, gripper_state=None):
        if self._arm_path is None:
            self._arm_path = []
        self._arm_path.append(np.array(arm_state))
        if gripper_state is not None:
            if self._gripper_path is None:
                self._gripper_path = []
            self._gripper_path.append(np.array(gripper_state))

    def append_segment(self, arm_states, gripper_state=None):
        for state in arm_states:
            self.append(state, gripper_state)

    def __len__(self):
        return len(self.arm_path)

    @property
    def arm_path_arr(self):
        return np.stack(self._arm_path)

    @property
    def gripper_path_arr(self):
        if self._gripper_path is None:
            return None
        assert len(self._gripper_path) == len(self._arm_path)
        return np.stack(self._gripper_path)

    @property
    def path_arr(self):
        return np.concatenate((self.arm_path_arr, self.gripper_path_arr), axis=1)

    def __add__(self, other):
        assert isinstance(other, Path)
        new_arm_path = deepcopy(self._arm_path) + deepcopy(other.arm_path)
        new_gripper_path = None
        if self._gripper_path is not None:
            assert other.gripper_path is not None
            new_gripper_path = deepcopy(self._gripper_path) + deepcopy(other.gripper_path)

        return __class__(arm_path=new_arm_path, gripper_path=new_gripper_path)


class ConfigurationPath(Path):
    def interpolate(self, resolutions):
        pass


class CartesianPath(Path):
    def interpolate(self, pos_resolution=0.01, orn_resolution=np.pi/16):
        new_path = CartesianPath()
        for i in range(len(self) - 1):
            pose1 = self.arm_path[i]
            pose2 = self.arm_path[i + 1]
            poses = list(PBU.interpolate_poses(pose1, pose2, pos_resolution, orn_resolution))
            gri = None if self.gripper_path is None else self.gripper_path[i + 1]
            new_path.append_segment(poses, gri)
        return new_path

    @property
    def arm_path_arr(self):
        raise NotImplementedError


def configuration_path_to_cartesian_path(planner_robot, conf_path):
    assert isinstance(planner_robot, PlannerRobot)
    pose_path = CartesianPath()
    for i in range(len(conf_path)):
        conf = conf_path.arm_path[i]
        planner_robot.reset_plannable_joint_positions(conf)
        pose = planner_robot.get_eef_position_orientation()
        gripper_state = None if conf_path.gripper_path is None else conf_path.gripper_path[i]
        pose_path.append(pose, gripper_state=gripper_state)
    return pose_path


def plan_skill_open_prismatic(
        planner,
        obstacles,
        approach_pose,
        reach_distance,
        retract_distance,
        joint_resolutions=None
):
    """
    plan skill for opening articulated object with prismatic joint (e.g., drawers)
    Args:
        planner (PlannerRobot): planner
        obstacles (list, tuple): a list obstacle ids
        approach_pose (tuple): a tuple for approaching pose
        reach_distance (float): distance for reach to grasp
        retract_distance (float): distance for retract to open
        joint_resolutions (list, tuple): motion planning joint-space resolution

    Returns:
        path (CartesianPath)
    """
    assert isinstance(planner, PlannerRobot)

    approach_confs = planner.plan_joint_path(
        target_pose=approach_pose, obstacles=obstacles, resolutions=joint_resolutions)
    path = ConfigurationPath()
    path.append_segment(approach_confs, gripper_state=GRIPPER_OPEN)

    pose_path = configuration_path_to_cartesian_path(planner, path)

    reach_pose = PBU.multiply(pose_path.arm_path[-1], ([reach_distance, 0, 0], PBU.unit_quat()))
    retract_pose = PBU.multiply(pose_path.arm_path[-1], ([-retract_distance, 0, 0], PBU.unit_quat()))
    pose_path.append(reach_pose, gripper_state=GRIPPER_OPEN)
    pose_path = pose_path.interpolate(pos_resolution=0.05, orn_resolution=np.pi/8)

    retract_path = CartesianPath()
    retract_path.append_segment([reach_pose]*10, gripper_state=GRIPPER_CLOSE)
    retract_path.append(retract_pose, gripper_state=GRIPPER_CLOSE)
    retract_path.append_segment([retract_pose] * 5, gripper_state=GRIPPER_CLOSE)
    retract_path.append_segment([retract_pose]*2, gripper_state=GRIPPER_OPEN)
    retract_path = retract_path.interpolate(pos_resolution=0.01, orn_resolution=np.pi/8)
    return pose_path + retract_path


def plan_skill_grasp(
        planner,
        obstacles,
        approach_pose,
        reach_distance,
        joint_resolutions=None
):
    approach_confs = planner.plan_joint_path(
        target_pose=approach_pose, obstacles=obstacles, resolutions=joint_resolutions)
    path = ConfigurationPath()
    path.append_segment(approach_confs, gripper_state=GRIPPER_OPEN)

    pose_path = configuration_path_to_cartesian_path(planner, path)

    reach_pose = PBU.multiply(pose_path.arm_path[-1], ([reach_distance, 0, 0], PBU.unit_quat()))


@contextmanager
def world_saved():
    saved_world = PBU.WorldSaver()
    yield
    saved_world.restore()


def inverse_kinematics(robot_bid, eef_link, plannable_joints, target_pose):
    movable_joints = PBU.get_movable_joints(robot_bid)  # all joints that will be calculated by inv kinematics
    plannable_joints_rel_index = [movable_joints.index(j) for j in plannable_joints]  # relative index we need to plan for

    conf = p.calculateInverseKinematics(robot_bid, eef_link, target_pose[0], target_pose[1])
    conf = [conf[i] for i in plannable_joints_rel_index]
    return conf


def plan_joint_path(
        robot_bid,
        eef_link_name,
        plannable_joint_names,
        target_pose,
        obstacles=(),
        attachments=(),
        resolutions=None):

    if resolutions is not None:
        assert len(plannable_joint_names) == len(resolutions)

    eef_link = PBU.link_from_name(robot_bid, eef_link_name)
    plannable_joints = PBU.joints_from_names(robot_bid, plannable_joint_names)
    conf = inverse_kinematics(robot_bid, eef_link, plannable_joints, target_pose)
    # plan collision-free path
    with world_saved():
        path = PBU.plan_joint_motion(
            robot_bid,
            plannable_joints,
            conf,
            obstacles=obstacles,
            resolutions=resolutions,
            attachments=attachments
        )
    return path


def interpolate_cartesian_path(pos_path, orn_path, pos_resolution, orn_resolution, allow_skip_waypoint=False):
    assert len(pos_resolution) == 3
    assert isinstance(orn_resolution, float)
    assert len(pos_path) == len(orn_path)
    assert len(pos_path[0]) == 3  # [x, y, z]
    assert len(orn_path[0]) == 4  # [x, y, z, w]

    interp_pos_path = []
    interp_orn_path = []

    for i in range(len(pos_path) - 1):
        pos1, pos2 = pos_path[i:i+2]
        orn1, orn2 = orn_path[i:i+2]


class Gripper(object):
    def __init__(
            self,
            joint_names=(),
            finger_link_names=(),
            joint_min=(0.00, 0.00),
            joint_max=(1., 1.)
    ):
        self.joint_min = joint_min
        self.joint_max = joint_max
        self.joint_names = joint_names
        self.finger_link_names = finger_link_names

        self.body_id = None
        self.filename = None

    def load(self, filename=None, body_id=None, scale=1.):
        self.filename = filename
        if self.filename is not None:
            self.body_id = p.loadURDF(
                self.filename, globalScaling=scale, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)
        else:
            self.body_id = body_id
        # change friction to simulate rubber
        for l in self.finger_links:
            p.changeDynamics(self.body_id, l, lateralFriction=10)

        # set to open
        for i, jointIndex in enumerate(self.joints):
            p.resetJointState(self.body_id, jointIndex, self.joint_max[i])

    @property
    def joints(self):
        return PBU.joints_from_names(self.body_id, self.joint_names)

    @property
    def finger_links(self):
        return [PBU.link_from_name(self.body_id, n) for n in self.finger_link_names]

    def get_joint_positions(self):
        return np.array([p.getJointState(self.body_id, j)[0] for j in self.joints])

    def grasp(self, force=100.):
        self.set_joint_positions(self.joint_min, force=force)
        # self._magic_grasp(target_id, target_link=-1, joint_type=p.JOINT_FIXED)

    def ungrasp(self):
        self.set_joint_positions(self.joint_max)
        # self._magic_ungrasp()

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

    def _magic_grasp(self, target_id, target_link=-1, joint_type=p.JOINT_FIXED):
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

    def _magic_ungrasp(self):
        if self.grasp_cid is not None:
            p.removeConstraint(self.grasp_cid)
            self.grasp_cid = None


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

    @property
    def eef_link_index(self):
        return PBU.link_from_name(self.body_id, self.eef_link_name)

    def get_eef_position_orientation(self):
        return p.getLinkState(self.body_id, self.eef_link_index)[:2]

    def get_eef_position(self):
        return self.get_eef_position_orientation()[0]

    def get_eef_orientation(self):
        return self.get_eef_position_orientation()[1]

    def set_eef_position_orientation(self, pos, orn):
        raise NotImplementedError

    def reset_eef_position_orientation(self, pos, orn):
        raise NotImplementedError

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
        conf = inverse_kinematics(self.body_id, self.eef_link_index, self.plannable_joints, target_pose=(pos, orn))
        self.reset_plannable_joint_positions(conf)

    def set_eef_position_orientation(self, pos, orn, force=100.):
        conf = inverse_kinematics(self.body_id, self.eef_link_index, self.plannable_joints, target_pose=(pos, orn))
        self.set_plannable_joint_positions(conf)

    def plan_joint_path(self, target_pose, obstacles, resolutions=None):
        return plan_joint_path(
            self.body_id,
            self.eef_link_name,
            self.plannable_joint_names,
            target_pose,
            obstacles=obstacles,
            resolutions=resolutions
        )

    def inverse_kinematics(self, target_pose):
        return inverse_kinematics(self.body_id, self.eef_link_index, self.plannable_joints, target_pose=target_pose)


class PlannerRobot(PlannableRobot):
    def __init__(self, eef_link_name, init_base_pose, arm, gripper=None, plannable_joint_names=None):
        super(PlannerRobot, self).__init__(
            eef_link_name=eef_link_name,
            init_base_pose=init_base_pose,
            arm=arm,
            gripper=gripper,
            plannable_joint_names=plannable_joint_names
        )
        self.ref_robot = None

    def setup(self, robot, obstacles, hide_planner=True):
        self.ref_robot = robot
        self.disable_collision_with((robot.body_id,) + obstacles)
        self.synchronize(robot)
        if hide_planner:
            for l in PBU.get_all_links(self.body_id):
                p.changeVisualShape(self.body_id, l, rgbaColor=(0, 0, 1, 0))

    def set_collision_with(self, others, collision):
        for other in others:
            for gl in PBU.get_all_links(other):
                for cgl in PBU.get_all_links(self.body_id):
                    p.setCollisionFilterPair(other, self.body_id, gl, cgl, collision)

    def disable_collision_with(self, others):
        self.set_collision_with(others, 0)

    def enable_collision_with(self, others):
        self.set_collision_with(others, 1)

    @contextmanager
    def collision_enabled_with(self, others):
        self.set_collision_with(others, 1)
        yield
        self.set_collision_with(others, 0)

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

    def plan_joint_path(self, target_pose, obstacles, resolutions=None):
        with self.collision_enabled_with(obstacles):
            path = plan_joint_path(
                self.body_id,
                self.eef_link_name,
                self.plannable_joint_names,
                target_pose,
                obstacles=obstacles,
                resolutions=resolutions
            )
        return path


class ConstraintActuatedRobot(Robot):
    def __init__(self, eef_link_name, init_base_pose, gripper):
        super(ConstraintActuatedRobot, self).__init__(
            eef_link_name=eef_link_name,
            init_base_pose=init_base_pose,
            gripper=gripper
        )
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

    def set_eef_position_orientation(self, pos, orn):
        p.changeConstraint(self.cid, pos, orn)


def main():
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)
    PBU.set_camera(45, -40, 2, (0, 0, 0))

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    cabinet_0007 = os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
    cabinet_0004 = os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')

    obj1 = InteractiveObj(filename=cabinet_0007)
    obj1.load()
    obj1.set_position([0,0,0.5])

    for jointIndex in range(p.getNumJoints(obj1.body_id)):
        friction = 0
        p.setJointMotorControl2(obj1.body_id, jointIndex, p.POSITION_CONTROL, force=friction)

    for l in range(p.getNumJoints(obj1.body_id)):
        p.changeDynamics(obj1.body_id, l, lateralFriction=10)

    obj2 = InteractiveObj(filename=cabinet_0004)
    obj2.load()
    obj2.set_position([0,0,2])

    obj3 = YCBObject('005_tomato_soup_can')
    obj3.load()
    obj3.set_position_orientation([0,0,1.2], [0, 0, 0, 1])
    for l in range(p.getNumJoints(obj3.body_id)):
        p.changeDynamics(obj3.body_id, l, lateralFriction=10)
    
    # gripper = ActuatedGripper(
    #     filename=os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper.urdf'),
    #     init_pose=([0, 0.3, 1.2], [0, 0, 0, 1]),
    #     eef_link_name="eef_link",
    #     gripper_joint_names=("left_gripper_joint", "right_gripper_joint")
    # )
    # gripper.load()
    # shadow_gripper = ShadowGripper.create_from(
    #     gripper,
    #     filename=os.path.join(gibson2.assets_path, 'models/grippers/cube_gripper/gripper.urdf'),
    #     plannable_joint_names=("txj", "tyj", "tzj", "rxj", "ryj", "rzj"),
    #     gripper_joint_names=(),
    # )

    # shadow_gripper = PlannableGripper.create_from(
    #     gripper,
    #     filename=os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper_plannable.urdf'),
    #     plannable_joint_names=("txj", "tyj", "tzj", "rxj", "ryj", "rzj"),
    #     gripper_joint_names=("left_gripper_joint", "right_gripper_joint")
    # )

    obstacles = (obj3.body_id, obj1.body_id, obj2.body_id)
    gripper = Gripper(
        joint_names=("left_gripper_joint", "right_gripper_joint"),
        finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
    )
    gripper.load(os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper.urdf'))
    robot = ConstraintActuatedRobot(
        eef_link_name="eef_link", init_base_pose=([0, 0.3, 1.2], [0, 0, 0, 1]), gripper=gripper)

    shadow_gripper = Gripper(
        joint_names=("left_gripper_joint", "right_gripper_joint"),
        finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
    )
    shadow_gripper.load(os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper_plannable.urdf'))    #
    arm = Arm(joint_names=("txj", "tyj", "tzj", "rxj", "ryj", "rzj"))
    arm.load(body_id=shadow_gripper.body_id)
    planner = PlannerRobot(
        eef_link_name="eef_link",
        init_base_pose=([0, 0.3, 1.2], [0, 0, 0, 1]),
        gripper=shadow_gripper,
        arm=arm,
        plannable_joint_names=arm.joint_names
    )
    planner.setup(robot, obstacles)

    target_pose = ([0.4379213,  0.0072391,  0.71218301], T.quaternion_multiply(T.quaternion_about_axis(np.pi, axis=[0, 0, 1]), T.quaternion_about_axis(np.pi / 2, axis=[1, 0, 0])))
    #
    # path = planner.plan_joint_path(target_pose=target_pose, obstacles=obstacles, resolutions=(0.25, 0.25, 0.25, 0.2, 0.2, 0.2))
    #
    # for conf in path:
    #     planner.reset_plannable_joint_positions(conf)
    #     for i in range(3):
    #         robot.set_eef_position_orientation(*planner.get_eef_position_orientation())
    #         p.stepSimulation()
    #         time.sleep(1./240.)

    path = plan_skill_open_prismatic(
        planner,
        obstacles=obstacles,
        approach_pose=target_pose,
        reach_distance=0.05,
        retract_distance=0.2,
        joint_resolutions=(0.25, 0.25, 0.25, 0.2, 0.2, 0.2)
    )

    for i in range(len(path)):
        pose = path.arm_path[i]
        grip = path.gripper_path[i]

        for _ in range(2):
            robot.set_eef_position_orientation(*pose)
            if grip == GRIPPER_CLOSE:
                robot.gripper.grasp()
            else:
                robot.gripper.ungrasp()
            p.stepSimulation()
            time.sleep(1. / 240.)

    # for i in range(10):
    #     robot.gripper.grasp()
    #     p.stepSimulation()
    #     time.sleep(1. / 240.)
    # #
    # pos = np.array(robot.get_eef_position())
    # for i in range(100):
    #     pos[0] += 0.002
    #     robot.set_eef_position(pos)
    #     p.stepSimulation()
    #     time.sleep(1. / 240.)

    planner.synchronize()

    pos = np.array(robot.get_eef_position())
    rot = np.array(robot.get_eef_orientation())
    grasped = False

    # rot = T.quaternion_about_axis(np.pi, [0, 0, 1])
    robot.set_eef_position_orientation(pos, rot)

    rot_yaw_pos = T.quaternion_about_axis(0.01, [0, 0, 1])
    rot_yaw_neg = T.quaternion_about_axis(-0.01, [0, 0, 1])
    rot_pitch_pos = T.quaternion_about_axis(0.01, [1, 0, 0])
    rot_pitch_neg = T.quaternion_about_axis(-0.01, [1, 0, 0])

    prev_key = None
    init_t = time.time()
    for i in range(24000):  # at least 100 seconds
        prev_rot = rot.copy()
        prev_pos = pos.copy()
        keys = p.getKeyboardEvents()
        # print((time.time() - init_t) / float(i + 1))
        print(pos - np.array(robot.get_eef_position()))

        p.stepSimulation()
        if ord('c') in keys and prev_key != keys:
            if grasped:
                gripper.ungrasp()
            else:
                gripper.grasp(obj3.body_id)
            grasped = not grasped

        if p.B3G_ALT in keys and p.B3G_LEFT_ARROW in keys:
            rot = T.quaternion_multiply(rot_yaw_pos, rot)
        if p.B3G_ALT in keys and p.B3G_RIGHT_ARROW in keys:
            rot = T.quaternion_multiply(rot_yaw_neg, rot)

        if p.B3G_ALT in keys and p.B3G_UP_ARROW in keys:
            rot = T.quaternion_multiply(rot_pitch_pos, rot)
        if p.B3G_ALT in keys and p.B3G_DOWN_ARROW in keys:
            rot = T.quaternion_multiply(rot_pitch_neg, rot)

        if p.B3G_ALT not in keys and p.B3G_LEFT_ARROW in keys:
            pos[1] -= 0.005
        if p.B3G_ALT not in keys and p.B3G_RIGHT_ARROW in keys:
            pos[1] += 0.005

        if p.B3G_ALT not in keys and p.B3G_UP_ARROW in keys:
            pos[0] -= 0.005
        if p.B3G_ALT not in keys and p.B3G_DOWN_ARROW in keys:
            pos[0] += 0.005

        if ord(',') in keys:
            pos[2] += 0.005
        if ord('.') in keys:
            pos[2] -= 0.005

        if not np.all(prev_pos == pos) or not np.all(prev_rot == rot):
            robot.set_eef_position_orientation(pos, rot)

        # p.saveBullet("/Users/danfeixu/workspace/igibson/states/{}.bullet".format(i))
        # print(jpos)
        time.sleep(1./240.)
        prev_key = keys

    p.disconnect()


if __name__ == '__main__':
    main()
