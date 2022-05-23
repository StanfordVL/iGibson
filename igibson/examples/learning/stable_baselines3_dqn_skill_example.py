# https://github.com/StanfordVL/behavior/blob/main/behavior/baselines/rl/stable_baselines3_ppo_training.py

import logging
import os
from typing import Callable

import igibson
from igibson.envs.skill_env import SkillEnv

log = logging.getLogger(__name__)

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import DQN, PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

except ModuleNotFoundError:
    log.error("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)


"""
Example training code using stable-baselines3 PPO for one BEHAVIOR activity.
Note that due to the sparsity of the reward, this training code will not converge and achieve task success.
This only serves as a starting point that users can further build upon.
"""


class CustomCombinedExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super(CustomCombinedExtractor, self).__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        feature_size = 128
        for key, subspace in observation_space.spaces.items():
            if key in ["proprioception", "task_obs"]:
                extractors[key] = nn.Sequential(nn.Linear(subspace.shape[0], feature_size), nn.ReLU())
            elif key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                n_input_channels = subspace.shape[2]  # channel last
                cnn = nn.Sequential(
                    nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[2], subspace.shape[0], subspace.shape[1]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["scan"]:
                n_input_channels = subspace.shape[1]  # channel last
                cnn = nn.Sequential(
                    nn.Conv1d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=0),
                    nn.ReLU(),
                    nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=0),
                    nn.ReLU(),
                    nn.Flatten(),
                )
                test_tensor = th.zeros([subspace.shape[1], subspace.shape[0]])
                with th.no_grad():
                    n_flatten = cnn(test_tensor[None]).shape[1]
                fc = nn.Sequential(nn.Linear(n_flatten, feature_size), nn.ReLU())
                extractors[key] = nn.Sequential(cnn, fc)
            elif key in ["occupancy_grid"]:
                continue
            else:
                raise ValueError("Unknown observation key: %s" % key)
            total_concat_size += feature_size

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> th.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key in ["occupancy_grid"]:
                continue
            if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2))
            elif key in ["scan"]:
                observations[key] = observations[key].permute((0, 2, 1))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        feature = th.cat(encoded_tensor_list, dim=1)
        # print('feature.shape: ', feature.shape)  #  torch.Size([1, 384])
        return feature


def main():
    # config_file = os.path.join('..', 'configs', "behavior_pick_and_place.yaml")
    # config_file = os.path.join('..', '..', 'configs', 'robots', "fetch_rl.yaml")
    config_file = os.path.join(igibson.configs_path, "fetch_rl.yaml")
    tensorboard_log_dir = "log_dir"
    prefix = "toggle_off"
    num_cpu = 1
    mode = "callback"  # 'callback'

    def make_env(rank: int, seed: int = 0) -> Callable:
        def _init() -> SkillEnv:
            env = SkillEnv(
                config_file=config_file,
                mode="headless",
                action_timestep=1 / 30.0,
                physics_timestep=1 / 300.0,
                print_log=False,
            )
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    env = VecMonitor(env)

    eval_env = SkillEnv(
        config_file=config_file,
        mode="headless",
        action_timestep=1 / 30.0,
        physics_timestep=1 / 300.0,
        print_log=False,
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model = DQN(
        "MultiInputPolicy",
        env,
        verbose=1,
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
        buffer_size=20 * 10,
    )

    log.debug(model.policy)

    if mode == "for_loop":
        for i in range(1000):
            model.learn(1000)

            mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
            log.info(f"After Training {i}: Mean reward: {mean_reward} +/- {std_reward:.2f}")

            model.save(os.path.join(tensorboard_log_dir, "ckpt_{:03d}_{}".format(i, prefix)))
        del model
    elif mode == "callback":
        # Save a checkpoint every 1000 steps
        checkpoint_callback = CheckpointCallback(save_freq=1000, save_path=tensorboard_log_dir, name_prefix=prefix)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(tensorboard_log_dir),
            log_path=os.path.join(tensorboard_log_dir),
            eval_freq=500,
        )
        # Create the callback list
        callback = CallbackList([checkpoint_callback, eval_callback])

        model.learn(1000000, callback=callback)

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
        log.info(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

        model.save(os.path.join(tensorboard_log_dir, "ckpt_{}".format(prefix)))
        del model

    model = PPO.load(os.path.join(tensorboard_log_dir, "ckpt_{}".format(prefix)))
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=20)
    log.info(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
