"""
Behavioral cloning agent network architecture and training
"""
import argparse
import glob

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Behavior_Dataset:
    def __init__(self, folder, task_name="assembling_gift_baskets"):
        demos = glob.glob(folder + task_name + "*")
        self.rgbs, self.proprioceptions, self.actions, self.task_obs = [], [], [], []
        print("Demo filenames", demos)
        for demo in demos:
            hf = h5py.File(demo)
            if len(self.actions) == 0:
                self.rgbs = np.asarray(hf["rgb"])
                self.proprioceptions = np.asarray(hf["proprioception"])
                self.actions = np.asarray(hf["action"])
                self.task_obs = np.asarray(hf["task_obs"])
            else:
                self.rgbs = np.append(self.rgbs, np.asarray(hf["rgb"]), axis=0)
                self.proprioceptions = np.append(self.proprioceptions, np.asarray(hf["proprioception"]), axis=0)
                self.actions = np.append(self.actions, np.asarray(hf["action"]), axis=0)
                self.task_obs = np.append(self.task_obs, np.asarray(hf["task_obs"]), axis=0)

    def to_device(self):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.rgbs = torch.tensor(self.rgbs, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        self.proprioceptions = torch.tensor(self.proprioceptions, dtype=torch.float32).to(device)
        self.actions = torch.tensor(self.actions, dtype=torch.float32).to(device)
        self.task_obs = torch.tensor(self.task_obs, dtype=torch.float32).to(device)
        print("shape of data:", self.proprioceptions.shape, self.actions.shape, self.task_obs.shape)


class BCNet_rgbp(nn.Module):
    """A behavioral cloning agent that uses RGB images and proprioception as state space"""

    def __init__(self, img_channels=3, proprioception_dim=20, num_actions=28):
        super(BCNet_rgbp, self).__init__()
        # image feature
        self.features1 = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.features2 = nn.Sequential(nn.Linear(proprioception_dim, 40), nn.ReLU())
        self.fc4 = nn.Linear(9216 + 40, 512)
        self.fc5 = nn.Linear(512, num_actions)

    def forward(self, imgs, proprioceptions):
        x1 = self.features1(imgs)
        x1 = self.flatten(x1)
        x2 = self.features2(proprioceptions)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc4(x)
        x = F.relu(x)
        x = self.fc5(x)
        return x


class BCNet_taskObs(nn.Module):
    def __init__(self, task_obs_dim=456, proprioception_dim=20, num_actions=28):
        super(BCNet_taskObs, self).__init__()
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train behavior cloning agent using BEHAVIOR human data")
    parser.add_argument("--demo_path", type=str, help="Directory of demo path")
    parser.add_argument(
        "--activity", type=str, help="Activitiy name defined in BEHAVIOR 100, e.g., assembling_gift_baskets"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    agb_data = Behavior_Dataset(args.demo_path, args.activity)
    agb_data.to_device()

    # Training
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bc_agent = BCNet_rgbp().to(device)
    bc_agent = BCNet_taskObs().to(device)
    optimizer = optim.Adam(bc_agent.parameters())

    NUM_EPOCH = 200
    PATH = "trained_models/" + task + ".pth"

    for epoch in range(NUM_EPOCH):
        optimizer.zero_grad()
        output = bc_agent(agb_data.task_obs, agb_data.proprioceptions)
        loss_func = nn.MSELoss()
        loss = loss_func(output, agb_data.actions)
        loss.backward()
        optimizer.step()

        print(loss.item())

    torch.save(bc_agent, PATH)
