from abc import abstractmethod, ABC
from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition
from gibson2.utils.utils import l2_distance


class ReachingGoal(BaseTerminationCondition):
    def __init__(self, config):
        super(ReachingGoal, self).__init__(config)
        self.dist_tol = self.config.get('dist_tol', 0.5)

    def get_termination(self, task, env):
        done = l2_distance(
            env.robots[0].get_end_effector_position(),
            task.target_pos) < self.dist_tol
        success = done
        return done, success
