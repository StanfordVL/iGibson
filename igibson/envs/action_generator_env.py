import gym

from igibson.action_generators.generator_base import ActionGeneratorError, BaseActionGenerator
from igibson.envs.igibson_env import iGibsonEnv


class ActionGeneratorEnv(gym.Env):
    def __init__(self, action_generator_class, reward_accumulation="sum", **kwargs):
        """
        @param action_generator_class: The BaseActionGenerator subclass to use for generating actions.
        @param reward_accumulation: Whether rewards across lower-level env timesteps should be summed or maxed.
        @param kwargs: The arguments to pass to the inner iGibsonEnv constructor.
        """
        self.env = iGibsonEnv(**kwargs)
        self.action_generator: BaseActionGenerator = action_generator_class(
            self.env.task, self.env.scene, self.env.robots[0]
        )

        self.action_space = self.action_generator.get_action_space()
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.reward_accumulation = reward_accumulation

    def step(self, action: int):
        # Run the goal generator and feed the goals into the motion planning env.
        accumulated_reward = 0

        state, done, info = None, None, None
        try:
            for action in self.action_generator.generate(action):
                state, reward, done, info = self.env.step(action)

                if self.reward_accumulation == "sum":
                    accumulated_reward += reward
                elif self.reward_accumulation == "max":
                    accumulated_reward = max(reward, accumulated_reward)
                else:
                    raise ValueError("Reward accumulation should be one of 'sum' and 'max'.")

                # If the episode is done, stop sending more commands.
                if done:
                    break
        except ActionGeneratorError as e:
            print(e)

        assert info is not None, "Action generator did not produce any actions."

        return self.env.get_state(), accumulated_reward, done, info

    def reset(self):
        return self.env.reset()

    @property
    def scene(self):
        return self.env.scene

    @property
    def task(self):
        return self.env.task

    @property
    def robots(self):
        return self.env.robots
