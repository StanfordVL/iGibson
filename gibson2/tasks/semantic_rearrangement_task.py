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
from collections import OrderedDict


class SemanticRearrangementTask(BaseTask):
    """
    Semantic Rearrangement Task
    The goal is to sort or gather multiple semantically distinct objects

    Args:
        env (BaseEnv): Environment using this task
        #objects (list of CustomWrappedObject): Object(s) to use for this task
        goal_pos (3-array): (x,y,z) cartesian global coordinates for the goal location
        randomize_initial_robot_pos (bool): whether to randomize initial robot position or not. If False,
            will selected based on pos_range specified in the config
    """

    def __init__(self, env, goal_pos=(0,0,0), randomize_initial_robot_pos=True):
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
        self.target_objects = self._create_objects(env=env)
        # Other internal vars
        self.randomize_initial_robot_pos = randomize_initial_robot_pos
        self.init_pos_range = np.array(self.config.get("pos_range", np.zeros((2,3))))
        self.init_rot_range = np.array(self.config.get("rot_range", np.zeros(2)))
        # Observation mode
        self.task_obs_format = self.config.get("task_obs_format", "global") # Options are global, egocentric
        assert self.task_obs_format in {"global", "egocentric"}, \
            f"Task obs format must be one of: [global, egocentric]. Got: {self.task_obs_format}"

    def _create_objects(self, env):
        """
        Helper function to create objects

        Returns:
            dict: objects mapped from name to BaseObject instances
        """
        objs = {}
        # Loop over all objects from the config file and load them
        for obj_config in self.config.get("objects", []):
            obj = CustomWrappedObject(env=env, **obj_config)
            # Import this object into the simulator
            env.simulator.import_object(obj=obj, class_id=obj.class_id)
            # Store a reference to this object
            objs[obj_config["name"]] = obj

        # Return created objects
        return objs

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
        for obj in self.target_objects.values():
            pos, ori = obj.sample_pose()
            obj.set_position_orientation(pos, ori)
        p.stepSimulation()

    def sample_initial_pose(self, env):
        """
        Sample robot initial pose

        :param env: environment instance
        :return: initial pose
        """
        if self.randomize_initial_robot_pos:
            _, initial_pos = env.scene.get_random_point(floor=self.floor_num)
        else:
            initial_pos = np.random.uniform(self.init_pos_range[0], self.init_pos_range[1])
        initial_orn = np.array([0, 0, np.random.uniform(self.init_rot_range[0], self.init_rot_range[1])])
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
        # Construct task obs -- consists of 3D locations of all target objects
        task_obs = OrderedDict()
        obs_cat = []
        # Loop over all target objects and add to our task obs
        for obj_name, obj in self.target_objects.items():
            obj_dist = np.array(obj.get_position())
            if self.task_obs_format == "egocentric":
                obj_dist -= np.array(env.robots[0].get_end_effector_position())
            task_obs[obj_name] = obj_dist
            obs_cat.append(obj_dist)
        # Add concatenated obs also
        task_obs["object-state"] = np.concatenate(obs_cat)

        return task_obs
