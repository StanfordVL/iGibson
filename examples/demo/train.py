import argparse
import os
import re
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from datetime import datetime
import torch.nn.functional as F
import torchvision.models as models
from dataset import InteractionDataset
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def normalize(array):
    return (array - np.min(array)) / (np.max(array) - np.min(array))

class FlowPredNet(nn.Module):
    def __init__(self):
        super(FlowPredNet, self).__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(3,16,kernel_size=5, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=5, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=5, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=5, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=5, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16,16,kernel_size=5, stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU()
            )
        self.fc = nn.Sequential(
            nn.Linear(2,14),
            nn.ReLU(),
            nn.Linear(14,28),
            nn.ReLU(),
            nn.Linear(28,128),
            nn.ReLU()
            )
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(in_channels=272,out_channels=32,kernel_size=5, stride=2,padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=5,stride=2,padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=5,stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=5,stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=5,stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=16,kernel_size=5,stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16,out_channels=3,kernel_size=6,stride=2, padding=1),
            )

        
    def forward(self, x, interaction):
        image_feat = self.convs(x).reshape(x.size(0), -1)
        interaction_feat = self.fc(interaction)
        concat_feat = torch.cat([image_feat, interaction_feat], 1)
        concat_feat = concat_feat.reshape(x.size(0), -1, 1, 1)
        out = self.deconvs(concat_feat)
        return out

if __name__ == "__main__":
    cudnn.benchmark = True

    d = InteractionDataset(filename="generated_data/test3000.pkl")
    dataloader = torch.utils.data.DataLoader(d, batch_size=16, shuffle=True, num_workers=4,
                                             drop_last=True, pin_memory=False)

    writer = SummaryWriter('generated_data/log2')

    
    net = FlowPredNet().cuda()
    weights_init(net)
    l1 = nn.L1Loss()

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0003, betas=(0.5, 0.999))
    n_iter = 0

    for epoch in range(1000):
        print(epoch)
        for i, data in enumerate(dataloader, 0):
            #print(data)
            optimizer.zero_grad()
            interaction = data[1].float().cuda() / 256.0
            image = data[0].cuda().permute(0,3,1,2)
            flow = data[2].cuda().permute(0,3,1,2)
            #print(torch.sum(flow,dim=(1,2,3)))
            pred = net(image,interaction)
            #print(torch.sum(pred, dim=(1,2,3)))
            loss = l1(pred, flow)
            print(loss)
            loss.backward()
            optimizer.step()
            n_iter += 1
            #print(loss, interaction)
            #pass
            if i == 0:
                img_grid = vutils.make_grid(image)
                writer.add_image('input_image', img_grid, n_iter)

                pred_grid = vutils.make_grid(pred)
                writer.add_image('pred', pred_grid, n_iter)

                flow_grid = vutils.make_grid(flow)
                writer.add_image('flow', flow_grid, n_iter)

            writer.add_scalar('Loss/train', loss.item(), n_iter)

            # if epoch > 100 and i == 0:
            #     plt.figure()
            #     plt.subplot(2,1,1)
            #     plt.imshow(normalize(flow[0].cpu().numpy().transpose(1,2,0)))
            #     plt.subplot(2,1,2)
            #     plt.imshow(normalize(pred.detach()[0].cpu().numpy().transpose(1,2,0)))
            #     plt.show()

    writer.close()