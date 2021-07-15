from igibson.termination_conditions.termination_condition_base import BaseTerminationCondition


class MaxCollision(BaseTerminationCondition):
    """
    MaxCollision used for navigation tasks
    Episode terminates if the robot has collided more than
    max_collisions_allowed times
    """

    def __init__(self, config):
        super(MaxCollision, self).__init__(config)
        self.max_collisions_allowed = self.config.get(
            'max_collisions_allowed', 500)

    def get_termination(self, task, env):
        """
        Return whether the episode should terminate.
        Terminate if the robot has collided more than self.max_collisions_allowed times

        :param task: task instance
        :param env: environment instance
        :return: done, info
        """
        done = env.collision_step > self.max_collisions_allowed
        success = False
        return done, success
