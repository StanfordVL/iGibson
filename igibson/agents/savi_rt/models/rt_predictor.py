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
from igibson.agents.savi_rt.models.audio_cnn import AudioCNN
from igibson.agents.savi_rt.models.smt_cnn import SMTCNN

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
        self.rt_map_input_size = 50
        self.rt_map_output_size = 28 #put it in self.config, also change in parallel_env 28
        self.rooms = 23

        max_out_scale = np.amax(self.out_scale)
        n_upscale = int(np.ceil(math.log(max_out_scale, 2)))
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
#         self.hidden_states = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)

        self.rnn = RNNStateEncoder(input_size=self.input_size, 
                                   hidden_size=self.hidden_size).to(self.device)
        self.outc = outconv(1, self.rooms)
        
       
    def init_hidden_states(self):
        hidden_states = torch.zeros(1, self.batch_size, self.hidden_size, device=self.device)
        return hidden_states

    def feature_alignment(self, local_feature_maps, pos, orn, masks, map_resolution):      
        batch_sz, channels, local_sz, _ = local_feature_maps.shape # batch_sz*step_sz, 8, 32, 32
        global_sz = self.rt_map_input_size # 50
        
        global_maps = np.zeros((batch_sz, channels, global_sz, global_sz))
        global_maps[:, :, int(global_sz/2-local_sz/2):int(global_sz/2+local_sz/2),
                        int(global_sz/2-local_sz/2):int(global_sz/2+local_sz/2)] = local_feature_maps
        transformed_global_maps = np.zeros((batch_sz, channels, global_sz, global_sz))
        
        for i in range(batch_sz):
            transformed = scipy.ndimage.rotate(global_maps[i], -90+orn[i]*180.0/3.141593, 
                                 axes=(1, 2), reshape=False)
            delta_pos = pos[i, :2]/map_resolution[i]
            transformed = scipy.ndimage.shift(transformed, [0, -delta_pos[1], delta_pos[0]])
            transformed_global_maps[i] = transformed
        # adding a weighting factor
        return transformed_global_maps
         
        
    def update(self, observations, dones, rt_hidden_states, visual_features=None, audio_features=None):
        # 23 rooms
        # save to observations
        with torch.no_grad():
            masks = torch.tensor([[0.0] if done else [1.0] for done in dones], 
                                 dtype=torch.float, device=self.device)
            global_maps_features, global_maps, rt_hidden_states = self.cnn_forward(observations, rt_hidden_states, masks)
            observations['rt_map_features'].copy_(torch.flatten(global_maps_features, start_dim=1)) 
            observations['rt_map'].copy_(global_maps)
        return rt_hidden_states
            
        
    def cnn_forward_visual(self, features):       
        x_feat = features.view(-1, 128, 1, 1) #[batch size*step size, 128, 1,1]
        for mod in self.scaler:
            x_feat = mod(x_feat)
        return x_feat

    def cnn_forward_audio(self, features):
        x_feat = self.audio_cnn(features).view(-1, 128, 1, 1)
        for mod in self.scaler:
            x_feat = mod(x_feat)
        return x_feat
    
    def cnn_forward(self, observations, rt_hidden_states, masks):
        
        visual_features = observations['visual_features']
        audio_features = observations['audio_features']
        
        local_vmaps = self.cnn_forward_visual(visual_features).cpu().detach().numpy()
        local_amaps = self.cnn_forward_audio(audio_features).cpu().detach().numpy()
        #[batch_sz*step_sz, 8, 32, 32]
        
        curr_poses = observations["pose_sensor"][:, :2].cpu().detach().numpy()
        curr_orns = observations["pose_sensor"][:, 2].cpu().detach().numpy()
        map_resolution = observations["map_resolution"].cpu().detach().numpy()
        
        global_vmaps = self.feature_alignment(local_vmaps, curr_poses, curr_orns, masks, map_resolution)
        global_amaps = self.feature_alignment(local_amaps, curr_poses, curr_orns, masks, map_resolution)
        # [batch_sz*step_sz, 8, 50, 50]
        
        global_vmaps = (torch.from_numpy(global_vmaps)).view(visual_features.shape[0], -1).to(self.device) 
        global_amaps = (torch.from_numpy(global_amaps)).view(visual_features.shape[0], -1).to(self.device)
        # [batch_sz*step_sz, 20000]

        global_maps = torch.stack([global_vmaps, global_amaps], dim=1).view(visual_features.shape[0], -1)
        # [batch_sz*step_sz, 2, 20000] (view)-> [batch_sz*step_sz, 2*20000]
        global_maps, rt_hidden_states = self.rnn(global_maps.type(torch.FloatTensor).to(self.device), 
                                               rt_hidden_states, masks)
        
        global_maps_features = global_maps.view(visual_features.shape[0], 1, 
                                       self.rt_map_output_size, self.rt_map_output_size)
        global_maps = self.outc(global_maps_features)
        return global_maps_features, global_maps, rt_hidden_states
        


