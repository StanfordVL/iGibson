from igibson.reward_functions.reward_function_base import BaseRewardFunction


class PotentialReward(BaseRewardFunction):
    """
    Potential reward
    Assume task has get_potential implemented; Low potential is preferred
    (e.g. a common potential for goal-directed task is the distance to goal)
    """

    def __init__(self, config):
        super(PotentialReward, self).__init__(config)
        self.potential_reward_weight = self.config.get(
            'potential_reward_weight', 1.0
        )

    def reset(self, task, env):
        """
        Compute the initial potential after episode reset

        :param task: task instance
        :param env: environment instance
        """
        self.potential = task.get_potential(env)

    def get_reward(self, task, env):
        """
        Reward is proportional to the potential difference between
        the current and previous timestep

        :param task: task instance
        :param env: environment instance
        :return: reward
        """
        new_potential = task.get_potential(env)
        reward = self.potential - new_potential
        reward *= self.potential_reward_weight
        self.potential = new_potential
        return reward
