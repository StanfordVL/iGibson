from contextlib import contextmanager
from copy import deepcopy
import numpy as np

import pybullet as p
import gibson2.external.pybullet_tools.transformations as T
import gibson2.external.pybullet_tools.utils as PBU


EEF_GRASP_FRAME = ((1, 0, 0), PBU.unit_quat())


class NoPlanException(Exception):
    pass


@contextmanager
def world_saved():
    saved_world = PBU.WorldSaver()
    yield
    saved_world.restore()


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
    """
    Convert a joint-space path to cartesian space path by computing forward dynamics with a planner robot
    Args:
        planner_robot: planner robot that we use to compute joint positions
        conf_path: configuration paths
    Returns: cartesian-space path
    """
    pose_path = CartesianPath()
    for i in range(len(conf_path)):
        conf = conf_path.arm_path[i]
        planner_robot.reset_plannable_joint_positions(conf)
        pose = planner_robot.get_eef_position_orientation()
        gripper_state = None if conf_path.gripper_path is None else conf_path.gripper_path[i]
        pose_path.append(pose, gripper_state=gripper_state)
    return pose_path


def inverse_kinematics(robot_bid, eef_link, plannable_joints, target_pose):
    """
    Compute inverse kinematics for an end effector pose wrt to a list of plannable joints
    Args:
        robot_bid (int): body id of the robot
        eef_link (int): link index of the end effector
        plannable_joints (tuple, list): a list of plannable joint index
        target_pose (tuple): (pos, orn) of the target eef pose

    Returns: a list of joint configurations
    """
    movable_joints = PBU.get_movable_joints(robot_bid)  # all joints that will be calculated by inv kinematics
    plannable_joints_rel_index = [movable_joints.index(j) for j in plannable_joints]

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
        resolutions=None
):

    if resolutions is not None:
        assert len(plannable_joint_names) == len(resolutions)

    eef_link = PBU.link_from_name(robot_bid, eef_link_name)
    plannable_joints = PBU.joints_from_names(robot_bid, plannable_joint_names)
    conf = inverse_kinematics(robot_bid, eef_link, plannable_joints, target_pose)
    if conf is None:
        raise NoPlanException("No IK Solution Found")

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
            attachments=attachments,
            restarts=10,
            iterations=100,
            smooth=50
        )
    if path is None:
        raise NoPlanException("No Plan Found")
    return path


def sample_positions_in_box(x_range, y_range, z_range):
    x_range = np.array(x_range)
    y_range = np.array(y_range)
    z_range = np.array(z_range)
    rand_pos = np.random.rand(3)
    rand_pos[0] = rand_pos[0] * (x_range[1] - x_range[0]) + x_range[0]
    rand_pos[1] = rand_pos[1] * (y_range[1] - y_range[0]) + y_range[0]
    rand_pos[2] = rand_pos[2] * (z_range[1] - z_range[0]) + z_range[0]
    return rand_pos


def compute_grasp_pose(object_frame, grasp_orientation, grasp_distance, grasp_frame=EEF_GRASP_FRAME):
    """
    Compute grasping pose within an @object_frame wrt to the world frame.
    Args:
        object_frame (tuple): object frame to compute grasp in
        grasp_orientation (tuple, list): quaternion of the grasping orientation
        grasp_distance (float): grasp distance relative to the object frame center
        grasp_frame (tuple): end effector grasping frame

    Returns: grasp pose in the world frame

    """
    pose = (PBU.unit_point(), grasp_orientation)
    transform = ((-grasp_distance * np.array(grasp_frame[0])).tolist(), grasp_frame[1])
    grasp_pose = PBU.multiply(object_frame, pose, transform)
    return grasp_pose
