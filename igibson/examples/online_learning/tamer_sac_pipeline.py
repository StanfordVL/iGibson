import os
import sys
from typing import Callable

import yaml
from human_feedback import HumanFeedback
from online_learning_interface import FeedbackInterface
from PyQt5.QtWidgets import *
from tamer_sac_core import SAC

import igibson
from igibson.envs.behavior_env import BehaviorEnv

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
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
            if key in ["rgb", "highlight", "depth", "seg", "ins_seg"]:
                observations[key] = observations[key].permute((0, 3, 1, 2))
            elif key in ["scan"]:
                observations[key] = observations[key].permute((0, 2, 1))
            encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return th.cat(encoded_tensor_list, dim=1)


def main():
    with open("configs/tamer_sac.yaml", "r") as f:
        config_data = yaml.load(f, Loader=yaml.FullLoader)

    config_file = config_data["env_config_file"]
    tensorboard_log_dir = config_data["tensorboard_log_dir"]
    env = BehaviorEnv(
        config_file=os.path.join("../configs/", config_file),
        mode="gui",
        action_timestep=1 / 30.0,
        physics_timestep=1 / 300.0,
        action_filter="all",
    )

    human_feedback = HumanFeedback()
    app = QApplication(sys.argv)
    feedback_gui = FeedbackInterface()

    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model = SAC(
        config_data["policy_name"],
        env,
        buffer_size=config_data["buffer_size"],
        verbose=config_data["verbose"],
        tensorboard_log=tensorboard_log_dir,
        policy_kwargs=policy_kwargs,
        human_feedback=human_feedback,
        save_every=config_data["save_every_steps"],
    )

    if config_data["load_model"] != 0:
        model = SAC.load(f"tamer_sac_{config_data['load_model']}.pt")

    model.learn(config_data["steps"], human_feedback_gui=feedback_gui)

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")

    model.save(f"tamer_sac_{config_data['steps']}.pt")
    del model

    model = SAC.load(f"tamer_sac_{config_data['steps']}.pt")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")


if __name__ == "__main__":
    main()
