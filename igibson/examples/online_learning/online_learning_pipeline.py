import os
import sys
from logging import Handler

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from human_feedback import HumanFeedback
from online_learning_interface import FeedbackInterface
from PyQt5.QtWidgets import *

import igibson
from igibson.envs.behavior_env import BehaviorEnv

app = None
feedback_gui = None


class OLNet_taskObs(nn.Module):
    def __init__(self, task_obs_dim=456, proprioception_dim=20, num_actions=11):
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


def train_ol_model(ol_agent, env, device, learning_rate):
    optimizer = None
    human_feedback = HumanFeedback()
    app = QApplication(sys.argv)
    feedback_gui = FeedbackInterface()
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
                obs, reward, done, info = env.step(a)
            else:
                reward = 0
            curr_keyboard_feedback = human_feedback.return_human_keyboard_feedback()
            if curr_keyboard_feedback:
                if "Pause" in str(curr_keyboard_feedback):
                    paused = not paused
                elif "Reset" in str(curr_keyboard_feedback):
                    obs = env.reset()
                    total_reward = 0
                    done = False
                    paused = False
                elif type(curr_keyboard_feedback) == list:
                    error = np.array(curr_keyboard_feedback) * learning_rate
                    label_action = torch.from_numpy(a + error).type(torch.FloatTensor).view(action.size()).to(device)
                    loss = nn.MSELoss()(action, label_action)
                    feedback_gui.updateLoss(loss.item())
                    loss.backward()
                    optimizer.step()
                    print(loss)
                else:
                    print(curr_keyboard_feedback)

            curr_mouse_feedback = human_feedback.return_human_mouse_feedback()
            if curr_mouse_feedback:
                print(curr_mouse_feedback)

            total_reward += reward
            feedback_gui.updateReward(reward)


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"
    iterations = 100
    ol_agent = None
    config_file = "behavior_full_observability_fetch.yaml"
    env = BehaviorEnv(
        config_file=os.path.join("../configs/", config_file),
        mode="gui_interactive",
        action_timestep=1 / 30.0,
        physics_timestep=1 / 300.0,
        action_filter="all",
    )

    train_ol_model(ol_agent, env, device, 0.1)
