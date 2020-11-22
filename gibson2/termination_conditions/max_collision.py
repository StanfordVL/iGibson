from abc import abstractmethod, ABC
from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition


class MaxCollision(BaseTerminationCondition):
    def __init__(self, config):
        super(MaxCollision, self).__init__(config)
        self.max_collisions_allowed = self.config.get(
            'max_collisions_allowed', 500)

    def get_termination(self, task, env):
        done = env.collision_step > self.max_collisions_allowed
        success = False
        return done, success
