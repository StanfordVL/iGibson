from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.tasks.semantic_rearrangement_task import SemanticRearrangementTask
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import numpy as np
import pybullet as p


class SemanticOrganizeAndFetch(iGibsonEnv):
    """
    This class corresponds to a reward-free semantic organize-and-fetch task, where the goal is to either sort objects
    from a pile and place them in semantically-meaningful locations, or to search for these objects and bring them
    to a specified goal location

    Args:
        config_file (dict or str): config_file as either a dict or filepath string
        task_mode (str): Take mode for this environment. Options are "organize" or "fetch"
        scene_id (str): override scene_id in config file
        mode (str): headless, gui, iggui
        action_timestep (float): environment executes action per action_timestep second
        physics_timestep (float): physics timestep for pybullet
        device_idx (int): which GPU to run the simulation and rendering on
        render_to_tensor (bool): whether to render directly to pytorch tensors
        automatic_reset (bool): whether to automatic reset after an episode finishes
    """
    def __init__(
        self,
        config_file,
        task_mode="organize",
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
    ):
        # Store other internal variables
        self.task = None
        self.task_mode = task_mode

        # Run super init
        super().__init__(
            config_file=config_file,
            scene_id=scene_id,
            mode=mode,
            action_timestep=action_timestep,
            physics_timestep=physics_timestep,
            device_idx=device_idx,
            render_to_tensor=render_to_tensor,
            automatic_reset=automatic_reset,
        )

    def load(self):
        """
        Load environment
        """
        # Make sure "task" in config isn't filled in, since we write directly to it here
        self.config["task"] = "semantic_rearrangement"

        # Run super call
        super().load()

    def load_task_setup(self):
        """
        Extends super call to make sure that self.task is the appropriate task for this env
        """
        super().load_task_setup()

        # Load task
        self.task = SemanticRearrangementTask(
            env=self,
            goal_pos=[0, 0, 0],
            randomize_initial_robot_pos=(self.task_mode == "fetch"),
        )

    def reset(self):
        """
        Reset the environment

        Returns:
            OrderedDict: state after reset
        """
        # Run super method
        state = super().reset()

        # Re-gather task obs since they may have changed
        if 'task_obs' in self.output:
            state['task_obs'] = self.task.get_task_obs(self)

        # Return state
        return state

    def get_state(self, collision_links=[]):
        """
        Extend super class to also add in proprioception

        :param collision_links: collisions from last physics timestep
        :return: observation as a dictionary
        """

        # Run super class call first
        state = super().get_state(collision_links=collision_links)

        if 'proprio' in self.output:
            # Add in proprio states
            state["proprio"] = self.robots[0].get_proprio_obs()

        return state

    def set_task_conditions(self, task_conditions):
        """
        Method to override task conditions (e.g.: target object), useful in cases such as playing back
            from demonstrations

        Args:
            task_conditions (dict): Keyword-mapped arguments to pass to task instance to set internally
        """
        self.task.set_conditions(task_conditions)

    def check_success(self):
        """
        Checks various success states and returns the keyword-mapped values

        Returns:
            dict: Success criteria mapped to bools
        """
        return self.task.check_success()
