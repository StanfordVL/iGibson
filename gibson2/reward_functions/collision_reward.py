from gibson2.reward_functions.reward_function_base import BaseRewardFunction


class CollisionReward(BaseRewardFunction):
    def __init__(self, config):
        super(CollisionReward, self).__init__(config)
        self.collision_reward_weight = self.config.get(
            'collision_reward_weight', -0.1
        )

    def get_reward(self, task, env):
        has_collision = float(len(env.collision_links) > 0)
        return has_collision * self.collision_reward_weight
