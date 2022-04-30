import logging

import numpy as np
import pybullet as p

from igibson.tasks.point_nav_fixed_task import PointNavFixedTask
from igibson.utils.utils import l2_distance, restoreState

log = logging.getLogger(__name__)


class PointNavFixedRandomTask(PointNavFixedTask):
    """
    Point Nav Random Task
    The goal is to navigate to a random goal position
    """

    def __init__(self, env):
        super(PointNavFixedRandomTask, self).__init__(env)
        self.target_pos = env.config["target_pos"]

    def reset_agent(self, env):
        """
        Reset robot initial pose.
        Sample initial pose and target position, check validity, and land it.
        :param env: environment instance
        """
        env.robots[0].reset()
        reset_success = False
        max_trials = 100

        # cache pybullet state
        # TODO: p.saveState takes a few seconds, need to speed up
        state_id = p.saveState()
        for i in range(max_trials):
            _, initial_pos = env.scene.get_random_point(floor=0)
            initial_orn = [0, 0, 0]
            reset_success = env.test_valid_position(env.robots[0], initial_pos, initial_orn)
            restoreState(state_id)
            if reset_success:
                break

        if not reset_success:
            print("WARNING: Failed to reset robot without collision")
            # TODO: setting the variables to None so that any code relying on these values fails
            # It is maybe better to create an assert here as we do not want to continue if we couldn't sample
            initial_pos = None
            initial_orn = None

        p.removeState(state_id)
        self.initial_pos = initial_pos
        self.initial_orn = initial_orn

        super(PointNavFixedRandomTask, self).reset_agent(env)
