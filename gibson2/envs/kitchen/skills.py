import numpy as np

from gibson2.envs.kitchen.plan_utils import ConfigurationPath, CartesianPath, configuration_path_to_cartesian_path
from gibson2.envs.kitchen.robots import GRIPPER_CLOSE, GRIPPER_OPEN
import gibson2.external.pybullet_tools.utils as PBU


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
        pour_angle,
        holding,
        joint_resolutions=DEFAULT_JOINT_RESOLUTIONS
):
    grasp_pose = PBU.multiply(PBU.invert(planner.ref_robot.get_eef_position_orientation()), PBU.get_pose(holding))
    target_pour_pose = PBU.end_effector_from_body(object_target_pose, grasp_pose)

    confs = planner.plan_joint_path(
        target_pose=target_pour_pose, obstacles=obstacles, resolutions=joint_resolutions, attachment_ids=(holding,))

    conf_path = ConfigurationPath()
    conf_path.append_segment(confs, gripper_state=GRIPPER_CLOSE)

    pour_path = configuration_path_to_cartesian_path(planner, conf_path)
    pour_path = pour_path.interpolate(pos_resolution=0.025, orn_resolution=np.pi / 4)
    pour_path.append_pause(30)
    return pour_path


class Skill(object):
    def __init__(self):
        pass


class SkillLibrary(object):
    def __init__(self, planner, obstacles):
        self.planner = planner
        self.obstacles = obstacles

    @property
    def skills(self):
        return [
            "grasp",
            "place",
            "pour"
        ]

    def grasp(self, ):
        pass