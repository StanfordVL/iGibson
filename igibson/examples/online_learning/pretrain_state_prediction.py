import argparse
import os

import numpy as np
import torch
import torch.optim as optim
from vanillaVAE import VanillaVAE

import igibson
from igibson.envs.behavior_env import BehaviorEnv

device = "cuda" if torch.cuda.is_available() else "cpu"

config_file = "behavior_full_observability_fetch.yaml"
env = BehaviorEnv(
    config_file=os.path.join(igibson.example_config_path, config_file),
    mode="gui",
    action_timestep=1 / 30.0,
    physics_timestep=1 / 300.0,
    action_filter="all",
)

obs = env.reset()

rgb_x = None
obs_proprioception_x = None
actions = None

state_predictor = VanillaVAE(in_channels=3, latent_dim=100, input_image_width=512, action_dim=11).to(device)
epochs = 100
batch_size = 8
optimizer = optim.SGD(state_predictor.parameters(), lr=0.001, momentum=0.9)


for epoch in range(epochs * batch_size):
    task_obs, proprioception, rgb = obs["task_obs"], obs["proprioception"], obs["rgb"]
    action = np.array(env.action_space.sample())
    # proprioception = torch.tensor(obs["proprioception"], dtype=torch.float32).unsqueeze(0).to(device)
    if rgb_x is not None:
        rgb_x = np.append(rgb_x, rgb[np.newaxis], axis=0)
        obs_proprioception_x = np.append(obs_proprioception_x, np.append(task_obs, proprioception)[np.newaxis], axis=0)
        actions = np.append(actions, action[np.newaxis], axis=0)
    else:
        rgb_x = rgb[np.newaxis]
        obs_proprioception_x = np.append(task_obs, proprioception)[np.newaxis]
        actions = action[np.newaxis]

    if len(rgb_x) == (batch_size + 1):
        optimizer.zero_grad()

        tensor_rgb = torch.tensor(rgb_x, dtype=torch.float32).to(device).permute((0, 3, 1, 2))
        tensor_rgb_x = tensor_rgb[:batch_size]
        tensor_rgb_y = tensor_rgb[1:]
        actions = torch.tensor(actions[:batch_size], dtype=torch.float32).to(device)

        prediction, _, mu, log_var = state_predictor(tensor_rgb_x, actions)
        loss = state_predictor.loss_function(prediction, tensor_rgb_y, mu, log_var)
        print(loss["loss"])
        loss = loss["loss"]
        loss.backward()
        optimizer.step()

        rgb_x = None
        obs_proprioception_x = None
        actions = None

    obs, _, done, _ = env.step(action)
    if done:
        obs = env.reset()
