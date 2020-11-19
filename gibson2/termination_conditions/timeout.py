from abc import abstractmethod, ABC
from gibson2.termination_conditions.termination_condition_base import BaseTerminationCondition


class Timeout(BaseTerminationCondition):
    def __init__(self, config):
        super(Timeout, self).__init__(config)
        self.max_step = self.config.get('max_step', 500)

    def get_termination(self, task, env):
        done = env.current_step >= self.max_step
        success = False
        return done, success
