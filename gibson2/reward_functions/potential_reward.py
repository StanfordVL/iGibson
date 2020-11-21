from gibson2.reward_functions.reward_function_base import BaseRewardFunction


class PotentialReward(BaseRewardFunction):
    def __init__(self, config):
        super(PotentialReward, self).__init__(config)
        self.potential_reward_weight = self.config.get(
            'potential_reward_weight', 1.0
        )

    def reset(self, task, env):
        self.potential = task.get_potential(env)

    def get_reward(self, task, env):
        new_potential = task.get_potential(env)
        reward = self.potential - new_potential
        reward *= self.potential_reward_weight
        self.potential = new_potential
        return reward
