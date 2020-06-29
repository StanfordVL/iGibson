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
from collections import OrderedDict


"""
Task plans -> skill parameters
Parameterized skill library
Skills + parameters -> joint-space motion plan
Motion plan -> task-space path
task-space path -> gripper actuation
"""
GRIPPER_OPEN = -1
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
        self._arm_path.append(arm_state)
        if gripper_state is not None:
            if self._gripper_path is None:
                self._gripper_path = []
            self._gripper_path.append(gripper_state)
        else:
            assert self._gripper_path is None

    def append_segment(self, arm_states, gripper_state=None):
        for state in arm_states:
            self.append(state, gripper_state)

    def append_pause(self, num_steps):
        for _ in range(num_steps):
            self._arm_path.append(deepcopy(self._arm_path[-1]))
            if self._gripper_path is not None:
                self._gripper_path.append(deepcopy(self._gripper_path[-1]))

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

        return self.__class__(arm_path=new_arm_path, gripper_path=new_gripper_path)


def interpolate_joint_positions(j1, j2, resolutions):
    j1 = np.array(j1)
    j2 = np.array(j2)
    resolutions = np.array(resolutions)
    num_steps = int(np.ceil(((j2 - j1) / resolutions).max()))
    new_confs = []
    for i in range(num_steps):
        fraction = float(i) / num_steps
        new_confs.append((1 - fraction) * j1 + fraction * j2)
    new_confs.append(j2)
    return new_confs


class ConfigurationPath(Path):
    def interpolate(self, resolutions):
        """Configuration-space interpolation"""
        new_path = ConfigurationPath()
        for i in range(len(self) - 1):
            conf1 = self.arm_path[i]
            conf2 = self.arm_path[i + 1]
            confs = interpolate_joint_positions(conf1, conf2, resolutions)
            gri = None if self.gripper_path is None else self.gripper_path[i + 1]
            new_path.append_segment(confs, gri)
        return new_path


class CartesianPath(Path):
    def get_delta_path(self):
        new_path = CartesianPath()
        for i in range(len(self) - 1):
            pose1 = self.arm_path[i]
            pose2 = self.arm_path[i + 1]
            pos_diff = tuple((np.array(pose2[0]) - np.array(pose1[0])).tolist())

            orn_diff = T.quaternion_multiply(pose2[1], T.quaternion_inverse(pose1[1]))
            orn_diff = tuple(orn_diff.tolist())
            gri = None if self.gripper_path is None else self.gripper_path[i + 1]
            new_path.append((pos_diff, orn_diff), gripper_state=gri)

        return new_path

    def compute_step_size(self):
        norms = []
        for i in range(len(self) - 1):
            pose1 = self.arm_path[i]
            pose2 = self.arm_path[i + 1]
            norms.append(np.linalg.norm(np.array(pose2[0]) - np.array(pose1[0])))
        return np.array(norms)

    def interpolate(self, pos_resolution=0.01, orn_resolution=np.pi/16):
        """Interpolate cartesian path. Quaternion is interpolated using slerp."""
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
        grasp_pose,
        retract_distance,
        reach_distance=0.,
        joint_resolutions=None
):
    """
    plan skill for opening articulated object with prismatic joint (e.g., drawers)
    Args:
        planner (PlannerRobot): planner
        obstacles (list, tuple): a list obstacle ids
        grasp_pose (tuple): pose for grasping the handle
        retract_distance (float): distance for retract to open
        reach_distance (float): distance for reaching to grasp
        joint_resolutions (list, tuple): motion planning joint-space resolution

    Returns:
        path (CartesianPath)
    """
    assert isinstance(planner, PlannerRobot)
    # approach handle
    approach_pose = PBU.multiply(grasp_pose, ([-reach_distance, 0, 0], PBU.unit_quat()))
    approach_confs = planner.plan_joint_path(
        target_pose=approach_pose, obstacles=obstacles, resolutions=joint_resolutions)
    conf_path = ConfigurationPath()
    conf_path.append_segment(approach_confs, gripper_state=GRIPPER_OPEN)
    grasp_path = configuration_path_to_cartesian_path(planner, conf_path)

    # grasp handle
    grasp_path.append(grasp_pose, gripper_state=GRIPPER_OPEN)
    grasp_path.append_segment([grasp_pose]*5, gripper_state=GRIPPER_CLOSE)
    grasp_path = grasp_path.interpolate(pos_resolution=0.05, orn_resolution=np.pi/8)

    # retract to open
    retract_pose = PBU.multiply(grasp_pose, ([-retract_distance, 0, 0], PBU.unit_quat()))
    retract_path = CartesianPath()
    retract_path.append(grasp_pose, gripper_state=GRIPPER_CLOSE)
    retract_path.append(retract_pose, gripper_state=GRIPPER_CLOSE)
    retract_path.append_pause(num_steps=5)  # pause before release
    retract_path.append_segment([retract_pose] * 2, gripper_state=GRIPPER_OPEN)
    retract_path = retract_path.interpolate(pos_resolution=0.01, orn_resolution=np.pi/8)
    return grasp_path + retract_path


def plan_skill_open_prismatic_joint(
        planner,
        obstacles,
        grasp_pose,
        retract_distance,
        reach_distance=0.,
        joint_resolutions=None
):
    """
    plan skill for opening articulated object with prismatic joint (e.g., drawers)
    Args:
        planner (PlannerRobot): planner
        obstacles (list, tuple): a list obstacle ids
        grasp_pose (tuple): pose for grasping the handle
        retract_distance (float): distance for retract to open
        reach_distance (float): distance for reaching to grasp
        joint_resolutions (list, tuple): motion planning joint-space resolution

    Returns:
        path (CartesianPath)
    """
    assert isinstance(planner, PlannerRobot)
    # approach handle
    approach_pose = PBU.multiply(grasp_pose, ([-reach_distance, 0, 0], PBU.unit_quat()))
    approach_confs = planner.plan_joint_path(
        target_pose=approach_pose, obstacles=obstacles, resolutions=joint_resolutions)
    conf_path = ConfigurationPath()
    conf_path.append_segment(approach_confs, gripper_state=GRIPPER_OPEN)
    conf_path.interpolate((0.05, 0.05, 0.05, 0.1, 0.1, 0.1))
    return conf_path


def plan_skill_grasp(
        planner,
        obstacles,
        grasp_pose,
        reach_distance=0.,
        lift_height=0.,
        joint_resolutions=None
):
    """
    plan skill for opening articulated object with prismatic joint (e.g., drawers)
    Args:
        planner (PlannerRobot): planner
        obstacles (list, tuple): a list obstacle ids
        grasp_pose (tuple): pose for grasping the object
        reach_distance (float): distance for reaching to grasp
        lift_height (float): distance to lift after grasping
        joint_resolutions (list, tuple): motion planning joint-space resolution

    Returns:
        path (CartesianPath)
    """

    reach_pose = PBU.multiply(grasp_pose, ([-reach_distance, 0, 0], PBU.unit_quat()))

    approach_confs = planner.plan_joint_path(
        target_pose=reach_pose, obstacles=obstacles, resolutions=joint_resolutions)
    conf_path = ConfigurationPath()
    conf_path.append_segment(approach_confs, gripper_state=GRIPPER_OPEN)

    grasp_path = configuration_path_to_cartesian_path(planner, conf_path)
    grasp_path.append(grasp_pose, gripper_state=GRIPPER_OPEN)
    grasp_path.append_segment([grasp_pose] * 5, gripper_state=GRIPPER_CLOSE)

    lift_pose = (grasp_pose[0][:2] + (grasp_pose[0][2] + lift_height,), grasp_pose[1])
    grasp_path.append(lift_pose, gripper_state=GRIPPER_CLOSE)

    return grasp_path.interpolate(pos_resolution=0.05, orn_resolution=np.pi / 8)


def plan_skill_place(
        planner,
        obstacles,
        place_pose,
        holding,
        joint_resolutions=None
):
    """
    plan skill for opening articulated object with prismatic joint (e.g., drawers)
    Args:
        planner (PlannerRobot): planner
        obstacles (list, tuple): a list obstacle ids
        place_pose (tuple): pose for grasping the object
        joint_resolutions (list, tuple): motion planning joint-space resolution

    Returns:
        path (CartesianPath)
    """
    obstacles = tuple(set(obstacles) - {holding})

    confs = planner.plan_joint_path(
        target_pose=place_pose, obstacles=obstacles, resolutions=joint_resolutions, attachments=(holding,))
    conf_path = ConfigurationPath()
    conf_path.append_segment(confs, gripper_state=GRIPPER_CLOSE)

    place_path = configuration_path_to_cartesian_path(planner, conf_path)
    place_path.append_segment([place_pose] * 5, gripper_state=GRIPPER_OPEN)

    return place_path.interpolate(pos_resolution=0.05, orn_resolution=np.pi / 8)


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
        attachment_ids=(),
        resolutions=None):

    if resolutions is not None:
        assert len(plannable_joint_names) == len(resolutions)

    eef_link = PBU.link_from_name(robot_bid, eef_link_name)
    plannable_joints = PBU.joints_from_names(robot_bid, plannable_joint_names)
    conf = inverse_kinematics(robot_bid, eef_link, plannable_joints, target_pose)
    attachments = []
    for aid in attachment_ids:
        attachments.append(PBU.create_attachment(robot_bid, eef_link, aid))
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


class ObjectBank(object):
    def __init__(self):
        self._objects = OrderedDict()

    def add_object(self, name, o):
        assert isinstance(name, str)
        assert name not in self._objects
        assert isinstance(o, Object)
        self._objects[name] = o

    @property
    def objects(self):
        return list(self._objects.values())

    @property
    def body_ids(self):
        return tuple([o.body_id for o in list(self._objects.values())])

    def __getitem__(self, name):
        return self._objects[name]

    def __len__(self):
        return len(self._objects)


def set_articulated_object_dynamics(obj):
    for jointIndex in PBU.get_joints(obj.body_id):
        friction = 0
        p.setJointMotorControl2(obj.body_id, jointIndex, p.POSITION_CONTROL, force=friction)
    set_friction(obj, friction=10.)


def set_friction(obj, friction=10.):
    for l in PBU.get_all_links(obj.body_id):
        p.changeDynamics(obj.body_id, l, lateralFriction=friction)


def pose_to_array(pose):
    assert len(pose) == 2
    assert len(pose[0]) == 3
    assert len(pose[1]) == 4
    return np.hstack((pose[0], pose[1]))


class BaseKitchenEnv(object):
    MAX_DPOS = 0.1
    MAX_DROT = np.pi / 8
    def __init__(self, robot_base_pose, use_planner=False, hide_planner=True):
        self._hide_planner = hide_planner
        self._robot_base_pose = robot_base_pose
        self.objects = ObjectBank()
        self.planner = None

        self._create_robot()
        self._create_env()
        if use_planner:
            self._create_planner()
        assert isinstance(self.robot, Robot)

    @property
    def action_dimension(self):
        """Action dimension"""
        return 7  # [x, y, z, ai, aj, ak, g]

    @property
    def name(self):
        """Environment name"""
        return "BasicKitchen"

    def _create_robot(self):
        gripper = Gripper(
            joint_names=("left_gripper_joint", "right_gripper_joint"),
            finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
        )
        gripper.load(os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper.urdf'))
        robot = ConstraintActuatedRobot(
            eef_link_name="eef_link", init_base_pose=self._robot_base_pose, gripper=gripper)

        self.robot = robot

    def _create_planner(self):
        shadow_gripper = Gripper(
            joint_names=("left_gripper_joint", "right_gripper_joint"),
            finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
        )
        shadow_gripper.load(
            os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper_plannable.urdf'),
            scale=1.1  # make the planner robot slightly larger than the real gripper to allow imprecise plan
        )
        arm = Arm(joint_names=("txj", "tyj", "tzj", "rxj", "ryj", "rzj"))
        arm.load(body_id=shadow_gripper.body_id)
        planner = PlannerRobot(
            eef_link_name="eef_link",
            init_base_pose=self._robot_base_pose,
            gripper=shadow_gripper,
            arm=arm,
            plannable_joint_names=arm.joint_names
        )
        planner.setup(self.robot, self.objects.body_ids, hide_planner=self._hide_planner)
        self.planner = planner

    def _create_env(self):
        floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
        p.loadMJCF(floor)

        drawer = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf'))
        drawer.load()
        drawer.set_position([0, 0, 0.5])
        set_articulated_object_dynamics(drawer)
        self.objects.add_object("drawer", drawer)

        cabinet = InteractiveObj(filename=os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf'))
        cabinet.load()
        cabinet.set_position([0, 0, 2])
        set_articulated_object_dynamics(cabinet)
        self.objects.add_object("cabinet", cabinet)

        can = YCBObject('005_tomato_soup_can')
        can.load()
        z = PBU.stable_z(can.body_id, drawer.body_id)
        can.set_position_orientation([0, 0, z], [0, 0, 0, 1])
        p.changeDynamics(can.body_id, -1, mass=1.0)
        set_friction(can)
        self.objects.add_object("can", can)

    def reset(self):
        self.robot.reset_base_position_orientation(*self._robot_base_pose)
        self.robot.reset()
        z = PBU.stable_z(self.objects["can"].body_id, self.objects["drawer"].body_id)
        self.objects["can"].set_position_orientation([0, 0, z], [0, 0, 0, 1])

    @property
    def sim_state(self):
        return np.zeros(3)  # TODO: implement this

    def step(self, action):
        assert len(action) == self.action_dimension
        action = action.copy()
        pos = action[:3]  # delta position
        orn = action[3:6]  # delta rotation in euler angle
        gri = action[-1]  # grasp or not
        pos *= self.MAX_DPOS
        orn *= self.MAX_DROT

        orn = T.quaternion_from_euler(*orn)
        self.robot.set_relative_eef_position_orientation(pos, orn)
        if gri > 0:
            self.robot.gripper.grasp()
        else:
            self.robot.gripper.ungrasp()
        for _ in range(2):
            p.stepSimulation()
            time.sleep(1. / 240.)

    def render(self, mode):
        """Render"""
        return

    def get_observation(self):
        # get proprio
        proprio = []
        gpose = self.robot.get_eef_position_orientation()
        proprio.append(np.array(self.robot.gripper.get_joint_positions()))
        proprio.append(pose_to_array(gpose))
        proprio = np.hstack(proprio).astype(np.float32)

        # get object info
        object_poses = []
        object_relative_poses = []
        for o in self.objects.objects:
            for l in PBU.get_all_links(o.body_id):
                lpose = PBU.get_link_pose(o.body_id, l)
                object_poses.append(pose_to_array(lpose))
                object_relative_poses.append(pose_to_array(PBU.multiply(lpose, PBU.invert(gpose))))

        object_poses = np.array(object_poses).astype(np.float32)
        object_relative_poses = np.array(object_relative_poses).astype(np.float32)

        return {
            "proprio": proprio,
            "object_poses": object_poses,
            "object_relative_poses": object_relative_poses,
            "object_positions": object_poses[:, :3],
            "object_relative_positions": object_relative_poses[:, :3]
        }

    def set_goal(self, **kwargs):
        """Set env target with external specification"""
        pass

    def is_done(self):
        """Check if the agent is done (not necessarily successful)."""
        return False

    def is_success(self):
        """Check if the task condition is reached."""
        can_position = self.objects["can"].get_position()
        drawer_aabb = PBU.get_aabb(self.objects["drawer"].body_id, 2)
        return PBU.aabb_contains_point(can_position, drawer_aabb)


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

    def reset(self):
        pass

    @property
    def eef_link_index(self):
        return PBU.link_from_name(self.body_id, self.eef_link_name)

    def reset_base_position_orientation(self, pos, orn):
        return p.resetBasePositionAndOrientation(self.body_id, pos, orn)

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

    def setup(self, robot, obstacle_ids, hide_planner=True):
        self.ref_robot = robot
        self.disable_collision_with((robot.body_id,) + obstacle_ids)
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

    def plan_joint_path(self, target_pose, obstacles, resolutions=None, attachments=(), synchronize=True):
        if synchronize:
            self.synchronize()  # synchronize planner with the robot

        with self.collision_enabled_with(obstacles):
            path = plan_joint_path(
                self.body_id,
                self.eef_link_name,
                self.plannable_joint_names,
                target_pose,
                obstacles=obstacles,
                resolutions=resolutions,
                attachment_ids=attachments
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
        p.changeConstraint(self.cid, maxForce=50000)
        self.set_eef_position_orientation(*self.get_eef_position_orientation())

    def reset_base_position_orientation(self, pos, orn):
        """Remove constraint before resetting the base to avoid messing up the simulation"""
        p.removeConstraint(self.cid)
        p.resetBasePositionAndOrientation(self.body_id, pos, orn)
        self._create_eef_constraint()

    def set_eef_position_orientation(self, pos, orn):
        p.changeConstraint(self.cid, pos, orn)


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


def pose_to_action(start_pose, target_pose, max_dpos, max_drot=None):
    action = np.zeros(6)
    action[:3] = target_pose[:3] - start_pose[:3]
    action[3:6] = T.euler_from_quaternion(T.quaternion_multiply(target_pose[3:], T.quaternion_inverse(start_pose[3:])))
    action[:3] = np.clip(action[:3] / max_dpos, -1., 1.)
    if max_dpos is not None:
        action[3:] = np.clip(action[3:] / max_drot, -1., 1.)
    if np.any(action == 1) or np.any(action == -1):
        print("WARNING: action is getting clipped")
    return action


def execute_planned_path(env, path):
    """Execute a planned path an relabel actions."""

    all_obs = []
    actions = []
    rewards = []
    states = []

    for i in range(len(path)):
        tpose = path.arm_path[i]
        grip = path.gripper_path[i]

        cpose = pose_to_array(env.robot.get_eef_position_orientation())
        tpose = pose_to_array(tpose)

        action = np.zeros(env.action_dimension)
        action[-1] = grip
        action[:-1] = pose_to_action(cpose, tpose, max_dpos=env.MAX_DPOS, max_drot=env.MAX_DROT)
        actions.append(action)

        rewards.append(float(env.is_success()))
        states.append(env.sim_state)
        all_obs.append(env.get_observation())

        env.step(action)

    all_obs.append(env.get_observation())
    actions.append(np.zeros(env.action_dimension))
    rewards.append(float(env.is_success()))
    states.append(env.sim_state)

    all_obs = dict((k, np.array([all_obs[i][k] for i in range(len(all_obs))])) for k in all_obs[0])
    return states, actions, rewards, all_obs


def get_demo(env):
    env.reset()
    all_states = []
    all_actions = []
    all_rewards = []
    all_obs = []

    drawer_grasp_pose = (
        [0.3879213,  0.0072391,  0.71218301],
        T.quaternion_multiply(T.quaternion_about_axis(np.pi, axis=[0, 0, 1]), T.quaternion_about_axis(np.pi / 2, axis=[1, 0, 0]))
    )
    path = plan_skill_open_prismatic(
        env.planner,
        obstacles=env.objects.body_ids,
        grasp_pose=drawer_grasp_pose,
        reach_distance=0.05,
        retract_distance=0.25,
        joint_resolutions=(0.25, 0.25, 0.25, 0.2, 0.2, 0.2)
    )
    states, actions, rewards, obs = execute_planned_path(env, path)
    all_states.append(states)
    all_actions.append(actions)
    all_rewards.append(rewards)
    all_obs.append(obs)

    can_grasp_pose = ((0.03, -0.005, 1.06), (0, 0, 1, 0))
    path = plan_skill_grasp(
        env.planner,
        obstacles=env.objects.body_ids,
        grasp_pose=can_grasp_pose,
        reach_distance=0.05,
        lift_height=0.05,
        joint_resolutions=(0.05, 0.05, 0.05, 0.2, 0.2, 0.2)
    )
    states, actions, rewards, obs = execute_planned_path(env, path)
    all_states.append(states)
    all_actions.append(actions)
    all_rewards.append(rewards)
    all_obs.append(obs)

    can_drop_pose = ((0.469, 0, 0.952), (0, 0, 1, 0))
    path = plan_skill_place(
        env.planner,
        obstacles=env.objects.body_ids,
        holding=env.objects["can"].body_id,
        place_pose=can_drop_pose,
        joint_resolutions=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    )
    states, actions, rewards, obs = execute_planned_path(env, path)
    all_states.append(states)
    all_actions.append(actions)
    all_rewards.append(rewards)
    all_obs.append(obs)

    all_states = np.concatenate(all_states, axis=0)
    all_actions = np.concatenate(all_actions, axis=0)
    all_rewards = np.concatenate(all_rewards, axis=0)
    all_obs = dict((k, np.concatenate([all_obs[i][k] for i in range(len(all_obs))], axis=0)) for k in all_obs[0])
    return all_states, all_actions, all_rewards, all_obs


def main():
    import argparse
    import h5py
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        required=True
    )
    parser.add_argument(
        "--n",
        type=int,
        default=10
    )
    args = parser.parse_args()

    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)
    PBU.set_camera(45, -40, 2, (0, 0, 0))
    env = BaseKitchenEnv(robot_base_pose=([0, 0.3, 1.2], [0, 0, 0, 1]), use_planner=True, hide_planner=False)

    if os.path.exists(args.file):
        os.remove(args.file)
    f = h5py.File(args.file)
    f_sars_grp = f.create_group("data")

    for i in range(args.n):
        states, actions, rewards, all_obs = get_demo(env)

        f_demo_grp = f_sars_grp.create_group("demo_{}".format(i))
        f_demo_grp.attrs["num_samples"] = (states.shape[0] - 1)
        f_demo_grp.create_dataset("states", data=states[:-1])
        f_demo_grp.create_dataset("actions", data=actions[:-1])
        for k in all_obs:
            f_demo_grp.create_dataset("obs/{}".format(k), data=all_obs[k][:-1])
            f_demo_grp.create_dataset("next_obs/{}".format(k), data=all_obs[k][1:])
    f.close()

    p.disconnect()


def interactive_session(env):
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)
    PBU.set_camera(45, -40, 2, (0, 0, 0))

    env.reset()
    robot = env.robot
    gripper = env.robot.gripper
    pos = np.array(robot.get_eef_position())
    rot = np.array(robot.get_eef_orientation())
    grasped = False

    rot_yaw_pos = T.quaternion_about_axis(0.01, [0, 0, 1])
    rot_yaw_neg = T.quaternion_about_axis(-0.01, [0, 0, 1])
    rot_pitch_pos = T.quaternion_about_axis(0.01, [1, 0, 0])
    rot_pitch_neg = T.quaternion_about_axis(-0.01, [1, 0, 0])

    prev_key = None
    for i in range(24000):  # at least 100 seconds
        print(env.is_success())

        prev_rot = rot.copy()
        prev_pos = pos.copy()
        keys = p.getKeyboardEvents()

        p.stepSimulation()
        if ord('c') in keys and prev_key != keys:
            if grasped:
                gripper.ungrasp()
            else:
                gripper.grasp()
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

        time.sleep(1./240.)
        prev_key = keys

    p.disconnect()


def test():
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)
    PBU.set_camera(45, -40, 2, (0, 0, 0))

    # floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    # p.loadMJCF(floor)
    #
    # cabinet_0007 = os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
    # cabinet_0004 = os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')
    #
    # obj1 = InteractiveObj(filename=cabinet_0007)
    # obj1.load()
    # obj1.set_position([0,0,0.5])
    #
    # for jointIndex in range(p.getNumJoints(obj1.body_id)):
    #     friction = 0
    #     p.setJointMotorControl2(obj1.body_id, jointIndex, p.POSITION_CONTROL, force=friction)
    #
    # for l in range(p.getNumJoints(obj1.body_id)):
    #     p.changeDynamics(obj1.body_id, l, lateralFriction=10)
    #
    # obj2 = InteractiveObj(filename=cabinet_0004)
    # obj2.load()
    # obj2.set_position([0,0,2])
    #
    # obj3 = YCBObject('005_tomato_soup_can')
    # obj3.load()
    # obj3.set_position_orientation([0,0,1.2], [0, 0, 0, 1])
    # for l in range(p.getNumJoints(obj3.body_id)):
    #     p.changeDynamics(obj3.body_id, l, lateralFriction=10)
    #
    # obstacles = (obj1.body_id, obj2.body_id, obj3.body_id)
    # gripper = Gripper(
    #     joint_names=("left_gripper_joint", "right_gripper_joint"),
    #     finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
    # )
    # gripper.load(os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper.urdf'))
    # robot = ConstraintActuatedRobot(
    #     eef_link_name="eef_link", init_base_pose=([0, 0.3, 1.2], [0, 0, 0, 1]), gripper=gripper)
    #
    # shadow_gripper = Gripper(
    #     joint_names=("left_gripper_joint", "right_gripper_joint"),
    #     finger_link_names=("left_gripper", "left_tip", "right_gripper", "right_tip")
    # )
    # shadow_gripper.load(os.path.join(gibson2.assets_path, 'models/grippers/basic_gripper/gripper_plannable.urdf'), scale=1.1)    #
    # arm = Arm(joint_names=("txj", "tyj", "tzj", "rxj", "ryj", "rzj"))
    # arm.load(body_id=shadow_gripper.body_id)
    # planner = PlannerRobot(
    #     eef_link_name="eef_link",
    #     init_base_pose=([0, 0.3, 1.2], [0, 0, 0, 1]),
    #     gripper=shadow_gripper,
    #     arm=arm,
    #     plannable_joint_names=arm.joint_names
    # )
    # planner.setup(robot, obstacles, hide_planner=False)
    # robot.set_eef_position_orientation(*robot.get_eef_position_orientation())
    #
    # for _ in range(100):
    #     p.stepSimulation()
    #
    # drawer_grasp_pose = (
    #     [0.3879213,  0.0072391,  0.71218301],
    #     T.quaternion_multiply(T.quaternion_about_axis(np.pi, axis=[0, 0, 1]), T.quaternion_about_axis(np.pi / 2, axis=[1, 0, 0]))
    # )
    #
    # planner.synchronize()
    # path = plan_skill_open_prismatic(
    #     planner,
    #     obstacles=obstacles,
    #     grasp_pose=drawer_grasp_pose,
    #     reach_distance=0.05,
    #     retract_distance=0.25,
    #     joint_resolutions=(0.25, 0.25, 0.25, 0.2, 0.2, 0.2)
    # )
    # for i in range(len(path)):
    #     pose = path.arm_path[i]
    #     grip = path.gripper_path[i]
    #
    #     for _ in range(2):
    #         robot.set_eef_position_orientation(*pose)
    #         if grip == GRIPPER_CLOSE:
    #             robot.gripper.grasp()
    #         else:
    #             robot.gripper.ungrasp()
    #         p.stepSimulation()
    #         time.sleep(1. / 240.)
    #
    # planner.synchronize()
    # can_grasp_pose = ((0.03, -0.005, 1.06), (0, 0, 1, 0))
    # path = plan_skill_grasp(
    #     planner,
    #     obstacles=obstacles,
    #     grasp_pose=can_grasp_pose,
    #     reach_distance=0.05,
    #     lift_height=0.05,
    #     joint_resolutions=(0.05, 0.05, 0.05, 0.2, 0.2, 0.2)
    # )
    #
    # for i in range(len(path)):
    #     pose = path.arm_path[i]
    #     grip = path.gripper_path[i]
    #
    #     for _ in range(2):
    #         robot.set_eef_position_orientation(*pose)
    #         if grip == GRIPPER_CLOSE:
    #             robot.gripper.grasp()
    #         else:
    #             robot.gripper.ungrasp()
    #         p.stepSimulation()
    #         time.sleep(1. / 240.)
    #
    # planner.synchronize()
    # can_drop_pose = ((0.469, 0, 0.952), (0, 0, 1, 0))
    # path = plan_skill_place(
    #     planner,
    #     obstacles=obstacles,
    #     holding=obj3.body_id,
    #     place_pose=can_drop_pose,
    #     joint_resolutions=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    # )
    #
    # for i in range(len(path)):
    #     pose = path.arm_path[i]
    #     grip = path.gripper_path[i]
    #
    #     for _ in range(2):
    #         robot.set_eef_position_orientation(*pose)
    #         if grip == GRIPPER_CLOSE:
    #             robot.gripper.grasp()
    #         else:
    #             robot.gripper.ungrasp()
    #         p.stepSimulation()
    #         time.sleep(1. / 240.)

    env = BaseKitchenEnv(robot_base_pose=([0, 0.3, 1.2], [0, 0, 0, 1]), use_planner=True, hide_planner=False)
    env.reset()
    drawer_grasp_pose = (
        [0.3879213,  0.0072391,  0.71218301],
        T.quaternion_multiply(T.quaternion_about_axis(np.pi, axis=[0, 0, 1]), T.quaternion_about_axis(np.pi / 2, axis=[1, 0, 0]))
    )

    path = plan_skill_open_prismatic(
        env.planner,
        obstacles=env.objects.body_ids,
        grasp_pose=drawer_grasp_pose,
        reach_distance=0.05,
        retract_distance=0.25,
        joint_resolutions=(0.25, 0.25, 0.25, 0.2, 0.2, 0.2)
    )
    execute_planned_path(env, path)
    can_grasp_pose = ((0.03, -0.005, 1.06), (0, 0, 1, 0))
    path = plan_skill_grasp(
        env.planner,
        obstacles=env.objects.body_ids,
        grasp_pose=can_grasp_pose,
        reach_distance=0.05,
        lift_height=0.05,
        joint_resolutions=(0.05, 0.05, 0.05, 0.2, 0.2, 0.2)
    )
    execute_planned_path(env, path)

    can_drop_pose = ((0.469, 0, 0.952), (0, 0, 1, 0))
    path = plan_skill_place(
        env.planner,
        obstacles=env.objects.body_ids,
        holding=env.objects["can"].body_id,
        place_pose=can_drop_pose,
        joint_resolutions=(0.05, 0.05, 0.05, 0.05, 0.05, 0.05)
    )
    execute_planned_path(env, path)

    # for i in range(len(path)):
    #     pose = path.arm_path[i]
    #     grip = path.gripper_path[i]
    #     action = np.hstack([pose[0], pose[1], grip])
    #     env.step(action)

    # pose = path.arm_path[0]
    # for i in range(len(delta_path)):
    #     # pose = path.arm_path[i]
    #     grip = delta_path.gripper_path[i]
    #     cpos, corn = pose
    #
    #     new_pos = tuple([cp + dp for cp, dp in zip(cpos, delta_path.arm_path[i][0])])
    #     new_orn = T.quaternion_multiply(delta_path.arm_path[i][1], corn)
    #     pose = (new_pos, new_orn)
    #     action = np.hstack([new_pos, new_orn, grip])
    #     env.step(action)

    # path = plan_skill_open_prismatic_joint(
    #     env.planner,
    #     obstacles=env.objects.body_ids,
    #     grasp_pose=drawer_grasp_pose,
    #     reach_distance=0.05,
    #     retract_distance=0.25,
    #     joint_resolutions=(0.1, 0.1, 0.1, 0.1, 0.1, 0.1)
    # )
    #
    # for i in range(len(path)):
    #     pose = path.arm_path[i]
    #     grip = path.gripper_path[i]
    #
    #     for _ in range(10):
    #         env.robot.arm.reset_joint_positions(pose)
    #         if grip == GRIPPER_CLOSE:
    #             env.robot.gripper.grasp()
    #         else:
    #             env.robot.gripper.ungrasp()
    #         p.stepSimulation()
    #         time.sleep(1. / 240.)


    # for i in range(10000):
    #     p.stepSimulation()
    #     time.sleep(1./240.)

    robot = env.robot
    gripper = env.robot.gripper
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
        print(env.is_success())

        prev_rot = rot.copy()
        prev_pos = pos.copy()
        keys = p.getKeyboardEvents()
        # print((time.time() - init_t) / float(i + 1))
        # print(pos - np.array(robot.get_eef_position()))
        # print(robot.get_eef_position_orientation())

        p.stepSimulation()
        if ord('c') in keys and prev_key != keys:
            if grasped:
                gripper.ungrasp()
            else:
                gripper.grasp()
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
