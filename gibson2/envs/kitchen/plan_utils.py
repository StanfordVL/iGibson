from contextlib import contextmanager
from copy import deepcopy
import numpy as np

import pybullet as p
import gibson2.external.pybullet_tools.transformations as T
import gibson2.external.pybullet_tools.utils as PBU
from gibson2.envs.kitchen.env_utils import pose_to_array, pose_to_action_euler, pose_to_action_axis_vector


EEF_GRASP_FRAME = ((1, 0, 0), PBU.unit_quat())


class NoPlanException(Exception):
    pass


class PreconditionNotSatisfied(NoPlanException):
    pass


@contextmanager
def world_saved():
    saved_world = PBU.WorldSaver()
    yield
    saved_world.restore()


class Path(object):
    def __init__(self, arm_path=None, gripper_path=None, gripper_collision=None, holding_collision=None):
        self._arm_path = arm_path
        self._gripper_path = gripper_path
        self._gripper_collision = gripper_collision
        self._holding_collision = holding_collision

    @property
    def arm_path(self):
        return self._arm_path

    @property
    def gripper_path(self):
        return self._gripper_path

    def append(self, arm_state, gripper_state):
        if self._arm_path is None:
            self._arm_path = []
        self._arm_path.append(arm_state)
        if self._gripper_path is None:
            self._gripper_path = []
        self._gripper_path.append(gripper_state)

    def append_segment(self, arm_states, gripper_state=None):
        for state in arm_states:
            self.append(state, gripper_state)

    def append_pause(self, num_steps):
        for _ in range(num_steps):
            self._arm_path.append(deepcopy(self._arm_path[-1]))
            self._gripper_path.append(deepcopy(self._gripper_path[-1]))

    def __len__(self):
        return len(self.arm_path)

    @property
    def arm_path_arr(self):
        return np.stack(self._arm_path)

    @property
    def gripper_path_arr(self):
        assert len(self._gripper_path) == len(self._arm_path)
        return np.stack(self._gripper_path)

    @property
    def path_arr(self):
        return np.concatenate((self.arm_path_arr, self.gripper_path_arr), axis=1)

    def __copy__(self):
        return self.__class__(arm_path=deepcopy(self._arm_path), gripper_path=deepcopy(self._gripper_path))

    def __deepcopy__(self, memodict={}):
        return self.__copy__()

    def __add__(self, other):
        assert isinstance(other, Path)
        # Hard copy if the other path is empty
        if other.arm_path is None and other.gripper_path is None:
            return self.__copy__()

        # otherwise concatenate the copied paths
        new_arm_path = deepcopy(self._arm_path) + deepcopy(other.arm_path)
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
            gri = self.gripper_path[i + 1]
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
            gri = self.gripper_path[i + 1]
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
            gri = self.gripper_path[i + 1]
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


def move_to_end_of_configuration_path(planner_robot, conf_path):
    """
    Move the planner robot to the end of a configuration path
    Args:
        planner_robot (PlannerRobot): planner robot
        conf_path (ConfigurationPath): configuration paths

    """
    planner_robot.reset_plannable_joint_positions(conf_path.arm_path[-1])


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
            iterations=50,
            smooth=30,
            max_distance=8e-3
        )
    if path is None:
        raise NoPlanException("No Motion Plan Found")
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


def execute_planned_path(env, path, noise=None, sleep_per_sim_step=0.0, store_full_trajectory=True, step_callback=None):
    """Execute a planned path an relabel actions."""

    # all_obs = []
    actions = []
    rewards = []
    states = []
    task_specs = []

    for i in range(len(path)):
        tpose = path.arm_path[i]
        grip = path.gripper_path[i]

        cpose = pose_to_array(env.robot.get_eef_position_orientation())
        tpose = pose_to_array(tpose)

        action = np.zeros(env.action_dimension)
        action[-1] = grip
        action[:-1] = pose_to_action_euler(cpose, tpose, max_dpos=env.MAX_DPOS, max_drot=env.MAX_DROT)
        # action[:-1] = pose_to_action_axis_vector(cpose, tpose, max_dpos=env.MAX_DPOS, max_drot=env.MAX_DROT)
        if noise is not None:
            assert len(noise) == (env.action_dimension - 1)
            noise_arr = np.array(noise)
            action[:6] += np.clip(np.random.randn(len(noise)) * noise_arr, -noise_arr * 2, noise_arr * 2)

        if store_full_trajectory or i == 0:
            actions.append(action)
            states.append(env.serialized_world_state)
            task_specs.append(env.task_spec)

        if step_callback is not None:
            step_callback(i)

        env.step(action, sleep_per_sim_step=sleep_per_sim_step, return_obs=False)
        if store_full_trajectory or i == 0:
            rewards.append(float(env.is_success()))

    if step_callback is not None:
        step_callback(len(path))
    # all_obs.append(env.get_observation())
    actions.append(np.zeros(env.action_dimension))
    rewards.append(float(env.is_success()))
    states.append(env.serialized_world_state)
    task_specs.append(env.task_spec)

    # all_obs = dict((k, np.array([all_obs[i][k] for i in range(len(all_obs))])) for k in all_obs[0])
    states = np.stack(states)
    actions = np.stack(actions)
    rewards = np.stack(rewards)
    task_specs = np.stack(task_specs)
    return {"states": states, "actions": actions, "rewards": rewards, "task_specs": task_specs}


def execute_planned_joint_path(env, path, noise=None, sleep_per_sim_step=0.0):
    """Execute a planned path an relabel actions."""

    # all_obs = []
    actions = []
    rewards = []
    states = []
    task_specs = []

    for i in range(len(path)):
        task_specs.append(env.task_spec)
        conf = path.arm_path[i]
        grip = path.gripper_path[i]

        action = np.zeros(env.action_dimension)
        action[-1] = grip
        action[:-1] = conf
        if noise is not None:
            assert len(noise) == (env.action_dimension - 1)
            noise_arr = np.array(noise)
            action[:6] += np.clip(np.random.randn(len(noise)) * noise_arr, -noise_arr * 2, noise_arr * 2)
        actions.append(action)
        states.append(env.serialized_world_state)

        env.step(action, sleep_per_sim_step=sleep_per_sim_step, return_obs=False)
        rewards.append(float(env.is_success()))

    # all_obs.append(env.get_observation())
    actions.append(np.zeros(env.action_dimension))
    rewards.append(float(env.is_success()))
    states.append(env.serialized_world_state)
    task_specs.append(env.task_spec)

    # all_obs = dict((k, np.array([all_obs[i][k] for i in range(len(all_obs))])) for k in all_obs[0])
    states = np.stack(states)
    actions = np.stack(actions)
    rewards = np.stack(rewards)
    task_specs = np.stack(task_specs)
    return {"states": states, "actions": actions, "rewards": rewards, "task_specs": task_specs}


def execute_skill(env, skill_lib, skill_params, target_object_id, skill_step, noise=None, sleep_per_sim_step=0.0):
    """
    Execute a skill given @skill_params and @target_object_id

    Try execute the skill. If skill fails in any way, record the exception and replace the path with an empty path
    execute_planned_path() will return a state_traj of length 1 (the current state) in this case.

    Args:
        env (BaseEnv): environment
        skill_lib (SkillLibrary): a library of skills to run
        skill_params (np.ndarray): an array of skill parameters
        target_object_id (int): target object body id in pybullet
        skill_step (int): current skill step in an episode
        noise (tuple): trajectory noise along each action dimension. None if no noise.
        sleep_per_sim_step (float): time to sleep in-between control steps.

    Returns:
        state_traj (dict): a dictionary of recorded trajectory
        skill_step (int): the next skill step
        skill_exception (NoPlanException): None of no exception is caught
    """
    skill_exception = None
    try:
        path = skill_lib.plan(params=skill_params, target_object_id=target_object_id)
    except NoPlanException as e:
        skill_exception = e
        path = CartesianPath(arm_path=[], gripper_path=[])
    state_traj = execute_planned_path(
        env, path, noise=noise, sleep_per_sim_step=sleep_per_sim_step, store_full_trajectory=False)

    # augment the trajectory with skill information
    traj_len = state_traj["states"].shape[0]

    # one-hot encoding for the object
    object_index_enc = np.zeros(len(env.objects))
    object_index_enc[env.objects.body_ids.index(target_object_id)] = 1
    object_index_enc_traj = np.tile(object_index_enc, (traj_len, 1))
    # skill parameters
    skill_params_traj = np.tile(skill_params, (traj_len, 1))
    # skill masks
    skill_param_masks = skill_lib.get_serialized_skill_param_mask(skill_params)
    skill_param_masks = np.tile(skill_param_masks, (traj_len, 1))
    # step index for the skill
    skill_step_traj = np.array([skill_step] * traj_len)
    # record whether skill succeeded or not
    skill_success = np.ones(traj_len) if skill_exception is None else np.zeros(traj_len)
    # mark the beginning of a skill in a trajectory
    skill_begin = np.zeros(traj_len)
    skill_begin[0] = 1

    state_traj["skill_step"] = skill_step_traj
    state_traj["skill_begin"] = skill_begin
    state_traj["skill_params"] = skill_params_traj
    state_traj["skill_param_masks"] = skill_param_masks
    state_traj["skill_object_index"] = object_index_enc_traj
    state_traj["skill_success"] = skill_success

    # record detailed skill params
    skill_param_dict = skill_lib.deserialize_skill_params(skill_params)
    skill_param_dict.update(skill_lib.get_skill_param_dict_metadata(skill_param_dict))
    for k, v in skill_param_dict.items():
        skill_param_dict[k] = np.tile(v, (traj_len, 1))

    state_traj["skill_param_dict"] = skill_param_dict

    exec_info = dict(
        exception=skill_exception
    )
    return state_traj, exec_info


class Buffer(object):
    def __init__(self):
        self.data = dict()

    def append(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)

    def aggregate(self):
        data = dict()
        for k, v in self.data.items():
            if isinstance(v[0], dict):
                data[k] = dict((kk, np.concatenate([v[i][kk] for i in range(len(v))], axis=0)) for kk in v[0])
            else:
                data[k] = np.concatenate(v, axis=0)
        # make sure all data items agree on sequence dimension
        v0 = list(data.values())[0]
        for v in data.values():
            if isinstance(v, dict):
                for vv in v.values():
                    assert v0.shape[0] == vv.shape[0]
            else:
                assert v.shape[0] == v0.shape[0]
        return data

    def __len__(self):
        return len(list(self.data.values())[0])
