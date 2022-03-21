import logging

import numpy as np
import pybullet as p

from igibson.tasks.point_nav_random_task import PointNavRandomTask
from igibson.reward_functions.reward_function_base import BaseRewardFunction
from igibson.reward_functions.potential_reward import PotentialReward
from igibson.reward_functions.collision_reward import CollisionReward
from igibson.reward_functions.point_goal_reward import PointGoalReward
from igibson.objects.visual_marker import VisualMarker
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
        self.target_obj  = None
        self.reward_functions = [
            PotentialReward(self.config), # geodesic distance, potential_reward_weight
            CollisionReward(self.config),
            PointGoalReward(self.config), # success_reward
            TimeReward(self.config), # time_reward_weight
        ]

        self.load_target(env)

    def reset_agent(self, env):
        super().reset_agent(env)
        self.target_obj.set_position(self.target_pos)
        audio_obj_id = self.target_obj.get_body_ids()[0]
        env.audio_system.registerSource(audio_obj_id, self.config['audio_dir'], enabled=True)
        env.audio_system.setSourceRepeat(audio_obj_id)

    def load_target(self, env):
        """
        Load target marker
        :param env: environment instance
        """

        cyl_length = 0.2

        self.target_obj = VisualMarker(
            visual_shape=p.GEOM_CYLINDER,
            rgba_color=[0, 0, 1, 0.3],
            radius=self.dist_tol,
            length=cyl_length,
            initial_offset=[0, 0, cyl_length / 2.0],
        )

        env.simulator.import_object(self.target_obj)

        # The visual object indicating the target location may be visible
        for instance in self.target_obj.renderer_instances:
            instance.hidden = False

