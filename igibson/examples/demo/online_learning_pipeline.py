import os
import threading as th
from logging import Handler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from human_feedback import HumanFeedback
from pynput import keyboard

import igibson
from igibson.envs.behavior_env import BehaviorEnv

human_feedback = None


class OLNet_taskObs(nn.Module):
    def __init__(self, task_obs_dim=456, proprioception_dim=20, num_actions=26):
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


def train_ol_model(ol_agent, env, device, learning_rate):
    global human_feedback
    optimizer = None
    feedback_dictionary = HumanFeedback().feedback_dictionary
    th.Thread(target=key_capture_thread, args=(), name="key_capture_thread", daemon=True).start()
    for _ in range(iterations):
        obs = env.reset()
        total_reward = 0
        done = False
        paused = False

        while not done:
            task_obs = torch.tensor(obs["task_obs"], dtype=torch.float32).unsqueeze(0).to(device)
            proprioception = torch.tensor(obs["proprioception"], dtype=torch.float32).unsqueeze(0).to(device)
            if not ol_agent:
                ol_agent = OLNet_taskObs(
                    task_obs_dim=task_obs.size()[-1], proprioception_dim=proprioception.size()[-1]
                ).to(device)
                ol_agent.train()
                optimizer = optim.Adam(ol_agent.parameters())
            optimizer.zero_grad()
            action = ol_agent(task_obs, proprioception)
            a = action.cpu().detach().numpy().squeeze(0)
            if not paused:
                obs, reward, done, _ = env.step(a)
            else:
                reward = 0

            if human_feedback:
                if "Press" in str(human_feedback):  # only use keypresses as reward signals
                    if human_feedback.key in feedback_dictionary:
                        feedback = [0 for _ in range(action.size()[-1])]
                        feedback = feedback_dictionary[human_feedback.key]
                        error = np.array(feedback) * learning_rate
                        label_action = (
                            torch.from_numpy(a + error).type(torch.FloatTensor).view(action.size()).to(device)
                        )
                        loss = 100 * nn.MSELoss()(action, label_action)
                        loss.backward()
                        optimizer.step()
                        print(loss)
                    elif human_feedback.key == keyboard.KeyCode.from_char("p"):  # use 'p' for pausing
                        paused = not paused
                    else:
                        print("Invalid feedback received")

                th.Thread(target=key_capture_thread, args=(), name="key_capture_thread", daemon=True).start()
                human_feedback = None
            total_reward += reward


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations = 100
    ol_agent = None
    config_file = "behavior_full_observability.yaml"
    env = BehaviorEnv(
        config_file=os.path.join(igibson.example_config_path, config_file),
        mode="headless",
        action_timestep=1 / 30.0,
        physics_timestep=1 / 300.0,
        action_filter="all",
    )

    train_ol_model(ol_agent, env, device, 0.1)
