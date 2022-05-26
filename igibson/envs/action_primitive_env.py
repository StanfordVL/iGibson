import logging
import time

import gym

from igibson.action_primitives.action_primitive_set_base import (
    REGISTERED_PRIMITIVE_SETS,
    ActionPrimitiveError,
    BaseActionPrimitiveSet,
)
from igibson.envs.igibson_env import iGibsonEnv

logger = logging.getLogger(__name__)


class ActionPrimitivesEnv(gym.Env):
    def __init__(
        self, action_primitives_class_name, reward_accumulation="sum", accumulate_obs=False, num_attempts=10, action_space_type='discrete', **kwargs
    ):
        """
        @param action_generator_class: The BaseActionPrimitives subclass name to use for generating actions.
        @param reward_accumulation: Whether rewards across lower-level env timesteps should be summed or maxed.
        @param accumulate_obs: Whether all observations should be returned instead of just the last lower-level step.
        @param num_attempts: How many times a primitive will be re-attempted if previous tries fail.
        @param kwargs: The arguments to pass to the inner iGibsonEnv constructor.
        """
        self.action_space_type = action_space_type
        self.env = iGibsonEnv(action_timestep=1 / 30.0, physics_timestep=1 / 120, **kwargs)
        self.action_generator: BaseActionPrimitiveSet = REGISTERED_PRIMITIVE_SETS[action_primitives_class_name](
            self.env, self.env.task, self.env.scene, self.env.robots[0], action_space_type=action_space_type,
        )

        self.action_space = self.action_generator.get_action_space()
        self.observation_space = self.env.observation_space
        self.reward_range = self.env.reward_range
        self.reward_accumulation = reward_accumulation
        self.accumulate_obs = accumulate_obs
        self.num_attempts = num_attempts

    def step(self, action: int, obs=None):
        # Run the goal generator and feed the goals into the env.
        accumulated_reward = 0
        accumulated_obs = []

        start_time = time.time()

        if self.action_space_type == 'multi_discrete':
            pre_action = [10, ]
        else:
            pre_action = 10
        for lower_level_action in self.action_generator.apply(pre_action, obs):
            obs, reward, done, info = self.env.step(lower_level_action)
            if self.accumulate_obs:
                accumulated_obs.append(obs)
            else:
                accumulated_obs = [obs]  # Do this to save some memory.

        for _ in range(self.num_attempts):
            # obs, done, info = None, None, {}
            # print('self.action_space_type: ', self.action_space_type)

            try:
                for lower_level_action in self.action_generator.apply(action, obs):
                    # print('lower_level_action: ', lower_level_action)
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
                break
            except ActionPrimitiveError as e:
                end_time = time.time()
                logger.error("AP time: {}".format(end_time - start_time))
                logger.warning("Action primitive failed! Exception {}".format(e))
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
        end_time = time.time()
        logger.error("action: {}, AP time: {}".format(action, end_time - start_time))
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
