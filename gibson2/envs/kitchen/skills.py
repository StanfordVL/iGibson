import numpy as np
from collections import OrderedDict

import gibson2.envs.kitchen.transform_utils as TU
import gibson2.external.pybullet_tools.transformations as T
from gibson2.envs.kitchen.plan_utils import ConfigurationPath, CartesianPath, configuration_path_to_cartesian_path, \
    compute_grasp_pose, NoPlanException
from gibson2.envs.kitchen.robots import GRIPPER_CLOSE, GRIPPER_OPEN
import gibson2.external.pybullet_tools.utils as PBU


ORIENTATIONS = OrderedDict({
    "front": T.quaternion_from_euler(0, 0, 0),
    "left": T.quaternion_from_euler(0, 0, np.pi / 2),
    "right": T.quaternion_from_euler(0, 0, -np.pi / 2),
    "back": T.quaternion_from_euler(0, 0, np.pi),
    "top": T.quaternion_from_euler(0, np.pi / 2, 0),
})

ORIENTATION_NAMES = list(ORIENTATIONS.keys())

DEFAULT_JOINT_RESOLUTIONS = (0.1, 0.1, 0.1, np.pi * 0.05, np.pi * 0.05, np.pi * 0.05)


def plan_move_to(
        planner,
        obstacles,
        target_pose,
        joint_resolutions=DEFAULT_JOINT_RESOLUTIONS
):
    confs = planner.plan_joint_path(
        target_pose=target_pose, obstacles=obstacles, resolutions=joint_resolutions)
    conf_path = ConfigurationPath()
    conf_path.append_segment(confs, gripper_state=GRIPPER_OPEN)
    path = configuration_path_to_cartesian_path(planner, conf_path)
    return path.interpolate(pos_resolution=0.05, orn_resolution=np.pi / 8)


def plan_skill_open_prismatic(
        planner,
        obstacles,
        grasp_pose,
        retract_distance,
        reach_distance=0.,
        joint_resolutions=DEFAULT_JOINT_RESOLUTIONS
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

    # retract a bit more to make room for the next motion
    retract_more_pose = PBU.multiply(retract_pose, ([-0.10, 0, 0], PBU.unit_quat()))
    retract_path.append(retract_more_pose, gripper_state=GRIPPER_OPEN)
    retract_path = retract_path.interpolate(pos_resolution=0.01, orn_resolution=np.pi/8)  # slowly open
    return grasp_path + retract_path


def plan_skill_grasp(
        planner,
        obstacles,
        grasp_pose,
        reach_distance=0.,
        lift_height=0.,
        grasp_speed=0.05,
        lift_speed=0.05,
        joint_resolutions=DEFAULT_JOINT_RESOLUTIONS,
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
    grasp_path = grasp_path.interpolate(pos_resolution=grasp_speed, orn_resolution=np.pi / 8)

    lift_path = CartesianPath()
    lift_path.append(grasp_pose, gripper_state=GRIPPER_CLOSE)
    lift_pose = (grasp_pose[0][:2] + (grasp_pose[0][2] + lift_height,), grasp_pose[1])
    lift_path.append(lift_pose, gripper_state=GRIPPER_CLOSE)
    lift_path = lift_path.interpolate(pos_resolution=lift_speed, orn_resolution=np.pi / 8)

    return grasp_path + lift_path


def plan_skill_place(
        planner,
        obstacles,
        object_target_pose,
        holding,
        retract_distance=0.,
        joint_resolutions=DEFAULT_JOINT_RESOLUTIONS
):
    """
    plan skill for placing an object at a target pose
    Args:
        planner (PlannerRobot): planner
        obstacles (list, tuple): a list obstacle ids
        object_target_pose (tuple): target pose for placing the object
        holding (int): object id of the object in hand
        retract_distance (float): how far should the gripper retract after placing
        joint_resolutions (list, tuple): motion planning joint-space resolution

    Returns:
        path (CartesianPath)
    """
    grasp_pose = PBU.multiply(PBU.invert(planner.ref_robot.get_eef_position_orientation()), PBU.get_pose(holding))
    target_place_pose = PBU.end_effector_from_body(object_target_pose, grasp_pose)

    confs = planner.plan_joint_path(
        target_pose=target_place_pose, obstacles=obstacles, resolutions=joint_resolutions, attachment_ids=(holding,))
    conf_path = ConfigurationPath()
    conf_path.append_segment(confs, gripper_state=GRIPPER_CLOSE)

    place_path = configuration_path_to_cartesian_path(planner, conf_path)
    place_path.append_pause(3)
    if retract_distance > 0:
        retract_pose = PBU.multiply(target_place_pose, ([-retract_distance, 0, 0], PBU.unit_quat()))
        place_path.append(target_place_pose, GRIPPER_OPEN)
        place_path.append(retract_pose, GRIPPER_OPEN)

    return place_path.interpolate(pos_resolution=0.05, orn_resolution=np.pi / 8)


def plan_skill_pour(
        planner,
        obstacles,
        object_target_pose,
        holding,
        pour_angle_speed=np.pi / 64,
        joint_resolutions=DEFAULT_JOINT_RESOLUTIONS
):
    grasp_pose = PBU.multiply(PBU.invert(planner.ref_robot.get_eef_position_orientation()), PBU.get_pose(holding))
    object_orn = PBU.get_pose(holding)[1]
    target_pose = PBU.end_effector_from_body((object_target_pose[0], object_orn), grasp_pose)
    target_pour_pose = PBU.end_effector_from_body(object_target_pose, grasp_pose)

    confs = planner.plan_joint_path(
        target_pose=target_pose, obstacles=obstacles, resolutions=joint_resolutions, attachment_ids=(holding,))

    conf_path = ConfigurationPath()
    conf_path.append_segment(confs, gripper_state=GRIPPER_CLOSE)

    approach_path = configuration_path_to_cartesian_path(planner, conf_path)
    approach_path = approach_path.interpolate(pos_resolution=0.025, orn_resolution=np.pi / 4)

    pour_path = CartesianPath()
    pour_path.append(target_pose, gripper_state=GRIPPER_CLOSE)
    pour_path.append(target_pour_pose, gripper_state=GRIPPER_CLOSE)
    pour_path = pour_path.interpolate(pos_resolution=0.025, orn_resolution=pour_angle_speed)
    return approach_path + pour_path


class Skill(object):
    def __init__(
            self,
            requires_holding=False,
            acquires_holding=False,
            releases_holding=False,
            joint_resolutions=DEFAULT_JOINT_RESOLUTIONS
    ):
        self.planner = None
        self.obstacles = None
        self.joint_resolutions = joint_resolutions
        self.requires_holding = requires_holding
        self.acquires_holding = acquires_holding
        self.releases_holding = releases_holding

    @property
    def action_dimension(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def plan(self, params, **kwargs):
        raise NotImplementedError

    def get_serialized_skill_params(self, **kwargs):
        raise NotImplementedError


class GraspDistOrn(Skill):
    def __init__(self, lift_height=0.1, lift_speed=0.05, joint_resolutions=DEFAULT_JOINT_RESOLUTIONS):
        super(GraspDistOrn, self).__init__(
            acquires_holding=True,
            requires_holding=False,
            releases_holding=False,
            joint_resolutions=joint_resolutions
        )
        self.lift_height = lift_height
        self.lift_speed = lift_speed

    @property
    def name(self):
        return "grasp_dist_orn"

    @property
    def action_dimension(self):
        return 4

    def plan(self, params, target_object_id=None):
        assert len(params) == self.action_dimension
        grasp_orn_quat = TU.axisangle2quat(*TU.vec2axisangle(params[1:]))
        grasp_pose = compute_grasp_pose(
            PBU.get_pose(target_object_id), grasp_orientation=grasp_orn_quat, grasp_distance=params[0])
        traj= plan_skill_grasp(
            planner=self.planner,
            obstacles=self.obstacles,
            grasp_pose=grasp_pose,
            joint_resolutions=self.joint_resolutions,
            lift_height=self.lift_height,
            lift_speed=self.lift_speed
        )
        return traj

    def get_serialized_skill_params(self, grasp_orn, grasp_distance):
        params = np.zeros(self.action_dimension)
        params[0] = grasp_distance
        params[1:] = TU.axisangle2vec(*TU.quat2axisangle(grasp_orn))
        return params


class GraspDistDiscreteOrn(GraspDistOrn):
    @property
    def name(self):
        return "grasp_dist_discrete_orn"

    @property
    def action_dimension(self):
        return len(ORIENTATIONS) + 1

    def plan(self, params, target_object_id=None):
        assert len(params) == self.action_dimension
        pose_idx = int(np.argmax(params[1:]))
        orn = ORIENTATIONS[ORIENTATION_NAMES[pose_idx]]
        grasp_pose = compute_grasp_pose(
            PBU.get_pose(target_object_id), grasp_orientation=orn, grasp_distance=params[0])
        traj= plan_skill_grasp(
            planner=self.planner,
            obstacles=self.obstacles,
            grasp_pose=grasp_pose,
            joint_resolutions=self.joint_resolutions,
            lift_height=self.lift_height,
            lift_speed=self.lift_speed
        )
        return traj

    def get_serialized_skill_params(self, grasp_orn_name, grasp_distance):
        params = np.zeros(self.action_dimension)
        orn_index = ORIENTATION_NAMES.index(grasp_orn_name)
        params[0] = grasp_distance
        params[1 + orn_index] = 1
        return params


class PlacePosOrn(Skill):
    def __init__(self, retract_distance=0.1, joint_resolutions=DEFAULT_JOINT_RESOLUTIONS, num_pause_steps=0):
        super(PlacePosOrn, self).__init__(
            requires_holding=True,
            releases_holding=True,
            acquires_holding=False,
            joint_resolutions=joint_resolutions
        )
        self.retract_distance = retract_distance
        self.num_pause_steps = num_pause_steps

    @property
    def name(self):
        return "place_pos_orn"

    @property
    def action_dimension(self):
        return 6

    def plan(self, params, target_object_id=None, holding_id=None):
        assert len(params) == self.action_dimension
        target_pos = np.array(PBU.get_pose(target_object_id)[0])
        target_pos[2] = PBU.stable_z(holding_id, target_object_id)
        target_pos += params[:3]
        orn = TU.axisangle2quat(*TU.vec2axisangle(params[3:]))
        place_pose = (target_pos, orn)

        traj = plan_skill_place(
            self.planner,
            obstacles=self.obstacles,
            object_target_pose=place_pose,
            holding=holding_id,
            retract_distance=self.retract_distance,
            joint_resolutions=self.joint_resolutions
        )
        traj.append_pause(self.num_pause_steps)
        return traj

    def get_serialized_skill_params(self, place_pos, place_orn):
        params = np.zeros(self.action_dimension)
        params[:3] = place_pos
        params[3:] = TU.axisangle2vec(*TU.quat2axisangle(place_orn))
        return params


class PlacePosDiscreteOrn(PlacePosOrn):
    @property
    def name(self):
        return "place_pos_discrete_orn"

    @property
    def action_dimension(self):
        return len(ORIENTATIONS) + 3

    def plan(self, params, target_object_id=None, holding_id=None):
        assert len(params) == self.action_dimension

        target_pos = np.array(PBU.get_pose(target_object_id)[0])
        target_pos[2] = PBU.stable_z(holding_id, target_object_id)
        target_pos += params[:3]

        orn_idx = int(np.argmax(params[3:]))
        orn = ORIENTATIONS[ORIENTATION_NAMES[orn_idx]]
        place_pose = (target_pos, orn)

        traj = plan_skill_place(
            self.planner,
            obstacles=self.obstacles,
            object_target_pose=place_pose,
            holding=holding_id,
            retract_distance=self.retract_distance,
            joint_resolutions=self.joint_resolutions
        )
        traj.append_pause(self.num_pause_steps)
        return traj

    def get_serialized_skill_params(self, place_pos, place_orn_name):
        params = np.zeros(self.action_dimension)
        orn_index = ORIENTATION_NAMES.index(place_orn_name)
        params[:3] = place_pos
        params[3 + orn_index] = 1
        return params


class PourPosOrn(Skill):
    def __init__(self, pour_angle_speed=np.pi / 64, joint_resolutions=DEFAULT_JOINT_RESOLUTIONS, num_pause_steps=30):
        super(PourPosOrn, self).__init__(
            requires_holding=True,
            acquires_holding=False,
            releases_holding=False,
            joint_resolutions=joint_resolutions
        )
        self.pour_angle_speed = pour_angle_speed
        self.num_pause_steps = num_pause_steps

    @property
    def name(self):
        return "pour_pos_orn"

    @property
    def action_dimension(self):
        return 6  # [x, y, z, ai, aj, az]

    def plan(self, params, target_object_id=None, holding_id=None):
        assert len(params) == self.action_dimension
        target_pos = np.array(PBU.get_pose(target_object_id)[0])
        target_pos += params[:3]
        orn = TU.axisangle2quat(*TU.vec2axisangle(params[3:]))
        pour_pose = (target_pos, orn)

        traj = plan_skill_pour(
            self.planner,
            obstacles=self.obstacles,
            object_target_pose=pour_pose,
            holding=holding_id,
            joint_resolutions=self.joint_resolutions,
            pour_angle_speed=self.pour_angle_speed
        )
        traj.append_pause(self.num_pause_steps)
        return traj

    def get_serialized_skill_params(self, pour_pos, pour_orn):
        params = np.zeros(self.action_dimension)
        params[:3] = pour_pos
        params[3:] = TU.axisangle2vec(*TU.quat2axisangle(pour_orn))
        return params


class PourPosAngle(PourPosOrn):
    @property
    def name(self):
        return "pour_pos_angle"

    @property
    def action_dimension(self):
        return 4  # [x, y, z, \theta]

    def plan(self, params, target_object_id=None, holding_id=None):
        assert len(params) == self.action_dimension
        target_pos = np.array(PBU.get_pose(target_object_id)[0])
        pour_pos = target_pos + params[:3]

        cur_orn = np.array(PBU.get_pose(holding_id)[1])
        angle = -params[3]
        xy_vec = target_pos[:2] - pour_pos[:2]
        perp_xy_vec = np.array([xy_vec[1], -xy_vec[0], 0])

        rot = T.quaternion_about_axis(angle, perp_xy_vec)
        orn = T.quaternion_multiply(rot, cur_orn)

        pour_pose = (pour_pos, orn)

        traj = plan_skill_pour(
            self.planner,
            obstacles=self.obstacles,
            object_target_pose=pour_pose,
            holding=holding_id,
            joint_resolutions=self.joint_resolutions,
            pour_angle_speed=self.pour_angle_speed
        )
        traj.append_pause(self.num_pause_steps)
        return traj

    def get_serialized_skill_params(self, pour_pos, pour_angle):
        params = np.zeros(self.action_dimension)
        params[:3] = pour_pos
        params[3] = pour_angle
        return params


class SkillLibrary(object):
    def __init__(self, planner, obstacles, skills):
        self.planner = planner
        self.obstacles = obstacles
        self._skills = skills
        for s in self.skills:
            s.planner = planner
            s.obstacles = obstacles
        self._holding = None

    @property
    def skills(self):
        return self._skills

    def reset(self):
        self._holding = None

    @property
    def skill_names(self):
        return [s.name for s in self._skills]

    def __len__(self):
        return len(self._skills)

    def name_to_skill_index(self, name):
        return self.skill_names.index(name)

    @property
    def action_dimension(self):
        return len(self.skills) + np.sum([s.action_dimension for s in self.skills])

    def _parse_serialized_skill_params(self, all_params):
        assert len(all_params) == self.action_dimension
        skill_index = int(np.argmax(all_params[:len(self.skills)]))
        assert skill_index < len(self.skills)
        ind = 0
        params = all_params[len(self.skills):]
        for i, s in enumerate(self.skills):
            if i == skill_index:
                return skill_index, params[ind:ind + s.action_dimension]
            else:
                ind += s.action_dimension
        return None

    def get_serialized_skill_params(self, skill_name, **kwargs):
        params = np.zeros(self.action_dimension)
        skill_index = self.name_to_skill_index(skill_name)
        skill_params = self.skills[skill_index].get_serialized_skill_params(**kwargs)
        params[skill_index] = 1

        ind = 0
        for i, s in enumerate(self.skills):
            if i == skill_index:
                params[len(self.skills) + ind: len(self.skills) + ind + s.action_dimension] = skill_params
            else:
                ind += s.action_dimension

        return params

    def plan(self, params, target_object_id):
        skill_index, skill_params = self._parse_serialized_skill_params(params)
        skill = self.skills[skill_index]
        # print(skill.name, skill_params, target_object_id)
        # print(skill.name)
        if skill.requires_holding:
            if self._holding is None:
                raise NoPlanException("Robot is not holding anything but is trying to run {}".format(skill.name))
            traj = skill.plan(skill_params, target_object_id=target_object_id, holding_id=self._holding)
            if skill.releases_holding:
                self._holding = None
        elif skill.acquires_holding:
            if self._holding is not None:
                raise NoPlanException("Robot is holding something but is trying to run {}".format(skill.name))
            traj = skill.plan(skill_params, target_object_id=target_object_id)
            self._holding = target_object_id
        else:
            traj = skill.plan(skill_params, target_object_id=target_object_id)
        return traj
