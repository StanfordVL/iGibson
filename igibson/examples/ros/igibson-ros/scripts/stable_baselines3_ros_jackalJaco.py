#!/usr/bin/python3

import rospy
import logging
import os
from typing import Callable
import yaml

import igibson
from igibson.utils.utils import parse_config
from igibson.envs.igibson_env import iGibsonEnv

try:
    import gym
    import torch as th
    import torch.nn as nn
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.preprocessing import maybe_transpose
    from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

except ModuleNotFoundError:
    print("stable-baselines3 is not installed. You would need to do: pip install stable-baselines3")
    exit(1)

"""
Example training code using stable-baselines3 PPO for PointNav task.
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


def main(selection="user", headless=False, short_exec=False):
    """
    Example to set a training process with Stable Baselines 3
    Loads a scene and starts the training process for a navigation task with images using PPO
    Saves the checkpoint and loads it again
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    config_file = "mobiman_jackal_jaco.yaml"
    config_path = igibson.ros_path + "/config/" + config_file
    config_data = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)

    config = parse_config(config_data)

    n_robot = config["n_robot"]
    mode = config["mode"]
    action_timestep = config["action_timestep"]
    physics_timestep = config["physics_timestep"]
    render_timestep = config["render_timestep"]
    use_pb_gui = config["use_pb_gui"]

    tensorboard_log_dir = igibson.ros_path + "/log"
    num_environments = n_robot

    print("[stable_baselines3_ros_jackalJaco::main] ros_path: " + str(igibson.ros_path))
    print("[stable_baselines3_ros_jackalJaco::main] config_file: " + str(config_file))
    print("[stable_baselines3_ros_jackalJaco::main] config_path: " + str(config_path))
    print("[stable_baselines3_ros_jackalJaco::main] config_data: " + str(config_data))
    
    print("[stable_baselines3_ros_jackalJaco::main] n_robot: " + str(n_robot))
    print("[stable_baselines3_ros_jackalJaco::main] mode: " + str(mode))
    print("[stable_baselines3_ros_jackalJaco::main] action_timestep: " + str(action_timestep))
    print("[stable_baselines3_ros_jackalJaco::main] physics_timestep: " + str(physics_timestep))
    print("[stable_baselines3_ros_jackalJaco::main] render_timestep: " + str(render_timestep))
    print("[stable_baselines3_ros_jackalJaco::main] use_pb_gui: " + str(use_pb_gui))
    
    print("[stable_baselines3_ros_jackalJaco::main] tensorboard_log_dir: " + str(tensorboard_log_dir))

    #print("[stable_baselines3_ros_jackalJaco::main] DEBUG_INF")
    #while 1:
    #    continue

    # Function callback to create environments
    def make_env(rank: int, seed: int = 0) -> Callable:
        def _init() -> iGibsonEnv:
            env = iGibsonEnv(
                config_file=config_path,
                mode=mode,
                action_timestep=1 / 10.0,
                physics_timestep=1 / 120.0,
                ros_node_id=rank,
                use_pb_gui=use_pb_gui
            )
            env.seed(seed + rank)
            return env

        set_random_seed(seed)
        return _init

    # Multiprocess
    env = SubprocVecEnv([make_env(i) for i in range(num_environments)])
    env = VecMonitor(env)

    '''
    # Create a new environment for evaluation
    eval_env = iGibsonEnv(
        config_file=os.path.join(igibson.configs_path, config_file),
        mode="headless",
        action_timestep=1 / 10.0,
        physics_timestep=1 / 120.0,
    )
    '''

    # Obtain the arguments/parameters for the policy and create the PPO model
    policy_kwargs = dict(
        features_extractor_class=CustomCombinedExtractor,
    )
    os.makedirs(tensorboard_log_dir, exist_ok=True)
    model = PPO("MultiInputPolicy", env, n_steps=512, verbose=1, tensorboard_log=tensorboard_log_dir, policy_kwargs=policy_kwargs)
    print(model.policy)

    '''
    print("BEFORE evaluate_policy 0")
    # Random Agent, evaluation before training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    print(f"Before Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    print("AFTER evaluate_policy 0")
    '''

    print("[stable_baselines3_ros_jackalJaco::main] BEFORE learn")
    # Train the model for the given number of steps
    total_timesteps = 100 if short_exec else 24000
    model.learn(total_timesteps, progress_bar=True)
    print("[stable_baselines3_ros_jackalJaco::main] AFTER learn")

    '''
    print("BEFORE evaluate_policy 1")
    # Evaluate the policy after training
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"After Training: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    print("AFTER evaluate_policy 1")

    # Save the trained model and delete it
    model.save("ckpt")
    del model

    # Reload the trained model from file
    model = PPO.load("ckpt")

    print("BEFORE evaluate_policy 2")
    # Evaluate the trained model loaded from file
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)
    print(f"After Loading: Mean reward: {mean_reward} +/- {std_reward:.2f}")
    print("AFTER evaluate_policy 2")
    '''

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()