import argparse
import os

import numpy as np
from pynput import keyboard
import threading as th
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import igibson
from igibson.envs.behavior_env import BehaviorEnv

device = "cuda" if torch.cuda.is_available() else "cpu"
human_feedback = None
iterations = 100
ol_agent = None
config_file = "behavior_full_observability.yaml"
env = BehaviorEnv(
    config_file=os.path.join(igibson.example_config_path, config_file),
    mode="gui",
    action_timestep=1 / 30.0,
    physics_timestep=1 / 300.0,
    action_filter = "all"
)

class OLNet_taskObs(nn.Module):
    def __init__(self, task_obs_dim = 456, proprioception_dim=20, num_actions=28):
        super(OLNet_taskObs, self).__init__()        
        # image feature
        self.fc1 = nn.Linear(task_obs_dim + proprioception_dim, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, num_actions)
        
    def forward(self, task_obs, proprioceptions):
        x = torch.cat((task_obs, proprioceptions), dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

def key_capture_thread():
    global human_feedback
    with keyboard.Events() as events:
        event = events.get(1e6)
        human_feedback = event

def train_ol_model(ol_agent, env, device):
    global human_feedback
    with torch.no_grad():
        th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
        for _ in range(iterations):
            obs = env.reset()
            total_reward = 0
            done = False
            while not(done):
                task_obs = torch.tensor(obs['task_obs'], dtype=torch.float32).unsqueeze(0).to(device)
                proprioception = torch.tensor(obs['proprioception'], dtype=torch.float32).unsqueeze(0).to(device)
                if not ol_agent:
                    ol_agent = OLNet_taskObs(task_obs_dim=task_obs.size()[-1], proprioception_dim=proprioception.size()[-1]).to(device)
                action = ol_agent(task_obs, proprioception)
                a = action.cpu().numpy().squeeze(0)
                a_no_reset = np.concatenate((a[:19], a[20:27]))
                obs, reward, done, info = env.step(a_no_reset)
                if human_feedback:
                    if 'Press' in str(human_feedback): # only use keypresses as reward signals
                        if human_feedback.key == keyboard.KeyCode.from_char('s'):
                            print("Negative feeddback received")
                        elif human_feedback.key == keyboard.KeyCode.from_char('d'):
                            print("Positive feedback received")
                        else:
                            print("Invalid feedback received")
                    th.Thread(target=key_capture_thread, args=(), name='key_capture_thread', daemon=True).start()
                    human_feedback = None
                total_reward += reward

train_ol_model(ol_agent, env, device)