from abc import abstractmethod, ABC


class BaseTask(ABC):
    def __init__(self, env):
        self.config = env.config
        self.reward_functions = []
        self.termination_conditions = []

    @abstractmethod
    def reset_scene(self, env):
        raise NotImplementedError()

    @abstractmethod
    def reset_agent(self, env):
        raise NotImplementedError()

    def get_reward(self, env, collision_links=[], action=None, info={}):
        reward = 0.0
        for reward_function in self.reward_functions:
            reward += reward_function.get_reward(self, env)

        return reward, info

    def get_termination(self, env, collision_links=[], action=None, info={}):
        done = False
        success = False
        for condition in self.termination_conditions:
            d, s = condition.get_termination(self, env)
            done = done or d
            success = success or s
        info['done'] = done
        info['success'] = success
        return done, info

    @abstractmethod
    def get_task_obs(self, env):
        raise NotImplementedError()

    def step(self, env):
        return
