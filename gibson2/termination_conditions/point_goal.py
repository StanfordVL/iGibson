from abc import abstractmethod, ABC
from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
from gibson2.utils.utils import l2_distance


class PointGoal(BaseTerminationCondition):
    def __init__(self, config):
        super(PointGoal, self).__init__(config)
        self.dist_tol = self.config.get('dist_tol', 0.5)

    def get_termination(self, env):
        done = l2_distance(
            env.robots[0].get_position()[:2],
            env.target_pos[:2]) < self.dist_tol
        success = done
        return done, success
