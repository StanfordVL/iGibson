from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition
from igibson.utils.utils import l2_distance


class ReachingGoal(BaseTerminationCondition):
    """
    ReachingGoal used for ReachingRandomTask
    Episode terminates if reaching goal is reached
    """

    def __init__(self, config):
        super(ReachingGoal, self).__init__(config)
        self.dist_tol = self.config.get('dist_tol', 0.5)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if reaching goal is reached (distance below threshold)

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = l2_distance(
            env.robots[0].get_end_effector_position(),
            task.target_pos) < self.dist_tol
        success = done
        return done, success
