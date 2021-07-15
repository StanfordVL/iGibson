from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition


class Timeout(BaseTerminationCondition):
    """
    Timeout
    Episode terminates if max_step steps have passed
    """

    def __init__(self, config):
        super(Timeout, self).__init__(config)
        self.max_step = self.config.get('max_step', 500)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if max_step steps have passed

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = env.current_step >= self.max_step
        success = False
        return done, success
