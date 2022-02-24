import logging

import numpy as np
import pybullet as p

from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.objects import cube
from igibson.utils.utils import l2_distance, restoreState


class TimeReward(BaseRewardFunction):
    """
    Time reward
    A negative reward per time step
    """

    def __init__(self, config):
        super().__init__(config)
        self.time_reward_weight = self.config.get(
            'time_reward_weight', -0.01)

    def get_reward(self, task, env):
        """
        Reward is proportional to the number of steps
        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        return self.time_reward_weight

class PointNavAVNavTask(PointNavRandomTask):
    """
    Redefine the task (reward functions)
    """
    def __init__(self, env):
        super().__init__(env)
        self.reward_functions = [
            PotentialReward(self.config), # geodesic distance, potential_reward_weight
            CollisionReward(self.config),
            PointGoalReward(self.config), # success_reward
            TimeReward(self.config), # time_reward_weight
        ]

    def reset_scene(self, env):
        super().reset_scene(env)
        source_location = self.target_pos
        self.audio_obj = cube.Cube(pos=source_location, dim=[0.05, 0.05, 0.05], 
                                    visual_only=False, 
                                    mass=0.5, color=[255, 0, 0, 1]) # pos initialized with default
        env.simulator.import_object(self.audio_obj)
        self.audio_obj_id = self.audio_obj.get_body_ids()[0]
        env.audio_system.registerSource(self.audio_obj_id, self.config['audio_dir'], enabled=True)
        env.audio_system.setSourceRepeat(self.audio_obj_id)
        env.simulator.attachAudioSystem(env.audio_system)

        env.audio_system.step()
