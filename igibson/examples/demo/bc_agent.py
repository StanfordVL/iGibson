import argparse
import os

import numpy as np
import torch
from simple_bc_agent import BCNet_rgbp, BCNet_taskObs

import igibson
from igibson.envs.behavior_env import BehaviorEnv


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the trained behvaior_cloning agent in behavior")
    parser.add_argument("--model", type=str, help="Path and filename of trained model")
    return parser.parse_args()


device = "cuda" if torch.cuda.is_available() else "cpu"
args = parse_args()
bc_agent = torch.load(args.model)
bc_agent.eval()

config_file = "behavior_onboard_sensing.yaml"
env = BehaviorEnv(
    config_file=os.path.join(igibson.example_config_path, config_file),
    mode="gui",
    action_timestep=1 / 30.0,
    physics_timestep=1 / 300.0,
    action_filter="all",
)

obs = env.reset()
total_reward = 0
done = False

with torch.no_grad():
    while not (done):
        task_obs = torch.tensor(obs["task_obs"], dtype=torch.float32).unsqueeze(0).to(device)
        proprioception = torch.tensor(obs["proprioception"], dtype=torch.float32).unsqueeze(0).to(device)
        action = bc_agent(task_obs, proprioception)
        a = action.cpu().numpy().squeeze(0)
        a_no_reset = np.concatenate((a[:19], a[20:27]))  # we do not allow reset action for agents here
        obs, reward, done, info = env.step(a_no_reset)
        total_reward += reward
        print(total_reward, info)
