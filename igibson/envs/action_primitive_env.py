import gym

from igibson.action_primitives.action_primitive_set_base import (
    REGISTERED_PRIMITIVE_SETS,
    ActionPrimitiveError,
    BaseActionPrimitiveSet,
)
from igibson.envs.igibson_env import iGibsonEnv


class ActionPrimitivesEnv(gym.Env):
    def __init__(
        self, action_primitives_class_name, reward_accumulation="sum", accumulate_obs=False, num_attempts=10, **kwargs
    ):
        """
        @param action_generator_class: The BaseActionPrimitives subclass name to use for generating actions.
        @param reward_accumulation: Whether rewards across lower-level env timesteps should be summed or maxed.
        @param accumulate_obs: Whether all observations should be returned instead of just the last lower-level step.
        @param num_attempts: How many times a primitive will be re-attempted if previous tries fail.
        @param kwargs: The arguments to pass to the inner iGibsonEnv constructor.
        """
        self.env = iGibsonEnv(**kwargs)
        self.action_generator: BaseActionPrimitiveSet = REGISTERED_PRIMITIVE_SETS[action_primitives_class_name](
            self.env.task, self.env.scene, self.env.robots[0]
        )

        self.action_space = self.action_generator.get_action_space()
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.reward_accumulation = reward_accumulation
        self.accumulate_obs = accumulate_obs
        self.num_attempts = num_attempts

    def step(self, action: int):
        # Run the goal generator and feed the goals into the motion planning env.
        accumulated_reward = 0
        accumulated_obs = []

        for _ in range(self.num_attempts):
            obs, done, info = None, None, {}
            try:
                for lower_level_action in self.action_generator.apply(action):
                    obs, reward, done, info = self.env.step(lower_level_action)

                    if self.reward_accumulation == "sum":
                        accumulated_reward += reward
                    elif self.reward_accumulation == "max":
                        accumulated_reward = max(reward, accumulated_reward)
                    else:
                        raise ValueError("Reward accumulation should be one of 'sum' and 'max'.")

                    if self.accumulate_obs:
                        accumulated_obs.append(obs)
                    else:
                        accumulated_obs = [obs]  # Do this to save some memory.

                    # Record additional info.
                    info["primitive_success"] = True
                    info["primitive_error_reason"] = None
                    info["primitive_error_metadata"] = None
                    info["primitive_error_message"] = None

                    # If the episode is done, stop sending more commands.
                    if done:
                        break
            except ActionPrimitiveError as e:
                # Record the error info.
                info["primitive_success"] = False
                info["primitive_error_reason"] = e.reason
                info["primitive_error_metadata"] = e.metadata
                info["primitive_error_message"] = str(e)

        # TODO: Think more about what to do when no observations etc. can be obtained.
        return_obs = None
        if accumulated_obs:
            if self.accumulate_obs:
                return_obs = accumulated_obs
            else:
                return_obs = accumulated_obs[-1]
        return return_obs, accumulated_reward, done, info

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
