#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from PIL import Image
import scipy

from igibson.agents.savi_rt.utils.utils import to_tensor
from igibson.agents.savi_rt.models.Unet_parts import UNetUp
from igibson.agents.savi_rt.models.rnn_state_encoder_rt import RNNStateEncoder


class SimpleWeightedCrossEntropy(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, output, target, meta=None, return_spatial=False):
        loss = F.cross_entropy(output, target, reduction='none')
        # If GT has padding, ignore the points that are not predictable
        if meta is not None and 'predictable_target' in meta:
            pred_index = meta['predictable_target'].long()
        else:
            pred_index = torch.ones_like(target).long()

        spatial_loss = loss
        # binary balanced: true
        loss = 0.5 * loss[pred_index * target > 0].mean() + 0.5 * loss[
            pred_index * (1 - target) > 0].mean()
        if return_spatial:
            return loss, spatial_loss
        return loss


class NonZeroWeightedCrossEntropy(SimpleWeightedCrossEntropy):
    def __init__(self):
        SimpleWeightedCrossEntropy.__init__(self)

    def forward(self, output, target, meta=None, return_spatial=False):
        if meta is not None and 'predictable_target' in meta:
            pred_index = meta['predictable_target']
        else:
            pred_index = torch.ones_like(target)

        pred_index = pred_index * (target > 0).long()
        target = target.type(torch.LongTensor).to(pred_index.device)
        output = output.type(torch.FloatTensor).to(pred_index.device)
        loss = F.cross_entropy(output, (target - 1) % target.max(),
                               reduction='none')
        spatial_loss = loss
        loss = loss[pred_index > 0].mean() if torch.sum(
            pred_index > 0) else torch.tensor(0.0).to(pred_index.device)
        if return_spatial:
            return loss, spatial_loss
        return loss

    
class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


class RTPredictor(nn.Module):
    def __init__(self, config, device, num_env=1): 
        super(RTPredictor, self).__init__()
        self.config = config
        self.device = device
        self.batch_size = num_env
 
        self.out_scale = (32, 32)
        self.n_channels_out = 128 # resnet 18
        self.rt_map_input_size = 50 #self.config["rt_map_size"] 140
        self.rt_map_output_size = 28 #put it in self.config, also change in parallel_env 28
        self.rooms = 23 # 23

        max_out_scale = np.amax(self.out_scale)
        n_upscale = int(np.ceil(math.log(max_out_scale, 2))) # = 5
        unet = [UNetUp(max(self.n_channels_out // (2**i), 8),
                   max(self.n_channels_out // (2**(i + 1)), 8),
                   bilinear=False,
                   norm="batchnorm") for i in range(n_upscale)]
        
        self.scaler = nn.ModuleList(unet).to(device)
        self.n_channels_out = max(self.n_channels_out // (2**(n_upscale)), 8)
        
        self.audio_cnn = nn.Sequential(
#             nn.BatchNorm1d(512),
            nn.Linear(128, 128), # changed from conv1d to linear  
        ).to(device)   
        
        # for encoder/decoder:

        self.input_size = 8*self.rt_map_input_size*self.rt_map_input_size*2
        self.hidden_size = self.rt_map_output_size*self.rt_map_output_size

        self.hidden_states = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)

        self.rnn = RNNStateEncoder(input_size=self.input_size, 
                                   hidden_size=self.hidden_size).to(self.device)
        self.outc = outconv(1, self.rooms)
        
        self.visual = SMTCNN(observations)
        self.audio = AudioCNN(observations, 128, "audio")


    def feature_alignment(self, local_feature_maps, curr_poses, curr_rpys):   
        # local feature maps: (batch, 64, 32, 32)
        # curr_poses: (batch, 3)
        # curr_rpys: (batch, 3)
        x, y, z = curr_poses[:,0], curr_poses[:,1], curr_poses[:,2]
        yaw = curr_rpys[:,2]
        batch_sz, rt_sz, local_sz, _ = local_feature_maps.shape
        global_sz = self.rt_map_input_size # 140
        global_unpadded_sz = int(global_sz/1.4) #self.config["rt_map_unpadded_size"] # 100
        padded_local_maps = np.zeros((batch_sz, rt_sz, global_sz, global_sz))
        # global sz: 140, local_sz: 32
        padded_local_maps[:, :, int(global_sz/2-local_sz/2):int(global_sz/2+local_sz/2),
                        int(global_sz/2-local_sz/2):int(global_sz/2+local_sz/2)] = local_feature_maps
        rotated_local_maps = np.zeros((batch_sz, rt_sz, global_sz, global_sz))
        for i in range(batch_sz):
            rot = scipy.ndimage.rotate(padded_local_maps[i], -90+yaw[i]*180.0/3.141593, 
                                 axes=(1, 2), reshape=False) # (64, 140, 140)
            rot = scipy.ndimage.shift(rot, [0, -y[i]/(global_unpadded_sz/1000), x[i]/(global_unpadded_sz/1000)]) # down, right
            rotated_local_maps[i] = rot
        # sanity check
        padded_img = Image.fromarray(padded_local_maps[0, 0].astype(np.uint8))
        rotated_local_map = padded_img.rotate(-90+yaw*180.0/3.141593, 
                                              translate=(x/(global_unpadded_sz/1000), -y/(global_unpadded_sz/1000))) 
                                                     #(right x, down y)

        return rotated_local_maps #(batch, 64, 140, 140)
        
    def init_hidden_states(self):
        self.hidden_states = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)
    
    def update(self, observations, dones, visual_features=None, audio_features=None): #step_observation, dones
        # 23 rooms
        if visual_features is None or audio_features is None:
            _, visual_features = self.visual(observations)
            _, audio_features = self.audio(observations)
        curr_poses = observations["pose_sensor"][:, :3].cpu().detach().numpy() #(9, 3)
        curr_rpys = observations["pose_sensor"][:, 3:6].cpu().detach().numpy() #(9, 3)
        
        local_vmaps = self.cnn_forward_visual(visual_features).cpu().detach().numpy() # batch,8,32,32  
        local_amaps = self.cnn_forward_audio(audio_features).cpu().detach().numpy() # batch,8,32,32 
        
        global_vmaps = self.feature_alignment(local_vmaps, curr_poses, curr_rpys)#(batch,8,rt_map_input_size,rt_map_input_size)
        global_amaps = self.feature_alignment(local_amaps, curr_poses, curr_rpys)
        
        global_vmaps = (torch.from_numpy(global_vmaps)).view(self.batch_size, -1).unsqueeze(0).to(self.device) 
        global_amaps = (torch.from_numpy(global_amaps)).view(self.batch_size, -1).unsqueeze(0).to(self.device)
        global_maps = torch.stack([global_vmaps, global_amaps], dim=2).view(1, self.batch_size, -1) #(1, batch, 2, 23*140*140)   
        #(1, batch, 2*global_vmaps[1:])   
        
        masks = torch.tensor([[0.0] if done else [1.0] for done in dones], 
                             dtype=torch.float, device=self.device)

        hidden_states_clone = self.hidden_states.clone().detach().requires_grad_(True)
        global_maps, self.hidden_states = self.rnn(global_maps.type(torch.FloatTensor).to(self.device), 
                                                   hidden_states_clone, masks)
        #(batch, 28*28)
        global_maps = global_maps.view(self.batch_size, 1, self.rt_map_output_size, self.rt_map_output_size)
        global_maps = self.outc(global_maps)
        #(batch, 23, 28, 28)
        global_maps = global_maps.permute(0, 2, 3, 1).view(self.batch_size, -1, self.rooms)
        return global_maps

    def cnn_forward_visual(self, features):
        # input feature size: batch_size * 64 * cnn_dims[0] * cnn_dims[1]       
        x_feat = features.view(self.batch_size, -1, 1, 1) # [1, 128, 1, 1]
        
        for mod in self.scaler:
            x_feat = mod(x_feat)
        # [1, 64, 32, 32]
        return x_feat

    def cnn_forward_audio(self, features):
        x_feat = self.audio_cnn(features).view(self.batch_size, -1, 1, 1)
        for mod in self.scaler:
            x_feat = mod(x_feat)
        # [1, 64, 32, 32]
        return x_feat
