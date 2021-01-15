from gibson2.tasks.task_base import BaseTask
import pybullet as p
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.termination_conditions.max_collision import MaxCollision
from gibson2.termination_conditions.timeout import Timeout
from gibson2.termination_conditions.out_of_bound import OutOfBound
from gibson2.reward_functions.null_reward import NullReward
from gibson2.objects.custom_wrapped_object import CustomWrappedObject

import logging
import numpy as np


class SemanticRearrangementTask(BaseTask):
    """
    Semantic Rearrangement Task
    The goal is to sort or gather multiple semantically distinct objects

    Args:
        env (BaseEnv): Environment using this task
        objects (list of CustomWrappedObject): Object(s) to use for this task
        goal_pos (3-array): (x,y,z) cartesian global coordinates for the goal location
        randomize_initial_robot_pos (bool): whether to randomize initial robot position or not. If False,
            will deterministically be set to @goal_pos instead
    """

    def __init__(self, env, objects, goal_pos=(0,0,0), randomize_initial_robot_pos=True):
        super().__init__(env)
        # Currently, this should be able to be done in either a gibson or igibson env
        # assert isinstance(env.scene, InteractiveIndoorScene), \
        #     'room rearrangement can only be done in InteractiveIndoorScene'
        self.termination_conditions = [
            MaxCollision(self.config),
            Timeout(self.config),
            OutOfBound(self.config),
        ]
        # Goal
        self.goal_pos = np.array(goal_pos)
        # Reward-free task currently
        self.reward_functions = [
            NullReward(self.config),
        ]
        self.floor_num = 0
        # Objects
        self.objects = objects
        # Other internal vars
        self.randomize_initial_robot_pos = randomize_initial_robot_pos

    def reset_scene(self, env):
        """
        Reset all scene objects as well as objects belonging to this task.

        :param env: environment instance
        """
        # Only reset scene objects if we're in an interactive scene
        if type(env.scene).__name__ == "InteractiveIndoorScene":
            env.scene.reset_scene_objects()
            env.scene.force_wakeup_scene_objects()

        # Reset objects belonging to this task specifically
        for obj in self.objects:
            pos, ori = obj.sample_pose()
            obj.set_position_orientation(pos, ori)

    def sample_initial_pose(self, env):
        """
        Sample robot initial pose

        :param env: environment instance
        :return: initial pose
        """
        if self.randomize_initial_robot_pos:
            _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        else:
            initial_pos = self.goal_pos
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose, check validity, and land it.

        :param env: environment instance
        """
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for _ in range(max_trials):
            initial_pos, initial_orn = self.sample_initial_pose(env)
            reset_success = env.test_valid_position(
                env.robots[0], initial_pos, initial_orn)
            p.restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            logging.warning("WARNING: Failed to reset robot without collision")

        env.land(env.robots[0], initial_pos, initial_orn)
        p.removeState(state_id)

        for reward_function in self.reward_functions:
            if reward_function is not None:
                reward_function.reset(self, env)

    def get_task_obs(self, env):
        """
        Get task-specific observation, including goal position, current velocities, etc.

        :param env: environment instance
        :return: task-specific observation
        """
        # TODO: Currently no task obs
        # task_obs = self.global_to_local(env, self.target_pos)[:2]
        # if self.goal_format == 'polar':
        #     task_obs = np.array(cartesian_to_polar(task_obs[0], task_obs[1]))

        # No task obs for now
        task_obs = None

        # linear velocity along the x-axis
        # linear_velocity = rotate_vector_3d(
        #     env.robots[0].get_linear_velocity(),
        #     *env.robots[0].get_rpy())[0]
        # # angular velocity along the z-axis
        # angular_velocity = rotate_vector_3d(
        #     env.robots[0].get_angular_velocity(),
        #     *env.robots[0].get_rpy())[2]
        # task_obs = np.append(
        #     task_obs, [linear_velocity, angular_velocity])

        return task_obs
