import gibson2
import json
import os
import math
import numpy as np
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import imageio
from PIL import Image

DEPTH_HIGH = 20.
DEPTH_CLIP_HIGH = 5.
DEPTH_CLIP_LOW = 0.
DEPTH_NOISE_RATE = 0.01

def get_label(image_path):
    img_dir = os.path.dirname(image_path)
    img_fname = os.path.splitext(os.path.basename(image_path))[0]
    img_id = img_fname.split('_')[1]
    json_path = os.path.join(img_dir, 'info_{}.json'.format(img_id))
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    moved = get_binary_label(data, image_path)
    x,y = data['interact_at']
    x = int(x / 4) # width 
    y = int(y / 4) # height
    return moved, (x,y) 

def get_binary_label(data, image_path):
    if data['hit'] is None:
        return False
    joint_pre = data['interaction_pre']['joint']
    joint_post = data['interaction_post']['joint']

    link_pre = data['interaction_pre']['link']
    link_post = data['interaction_post']['link']
    
    link_pos_delta = np.linalg.norm(np.array(link_pre[0]) 
                                  - np.array(link_post[0]))
    if link_pos_delta > 0.2:
        return True

    if joint_pre is not None:
        if joint_pre['type'] == 4:
            return False
        if joint_pre['type'] == 0:
            return abs(joint_post['pos'] - joint_pre['pos']) > 0.1
        if joint_pre['type'] == 1:
            return abs(joint_post['pos'] - joint_pre['pos']) > 0.1

    return link_pos_delta > 0.02

def get_depth_image(image_path):
    depth_path = image_path.replace('_rgb.png', '_3d.png')
    depth = np.array(
            imageio.imread(depth_path), np.float32
            ) * DEPTH_HIGH / float(np.iinfo(np.uint16).max)
    depth[depth > DEPTH_CLIP_HIGH] = 0.0
    depth[depth < DEPTH_CLIP_LOW] = 0.0
    depth /= DEPTH_CLIP_HIGH
    depth = add_naive_noise_to_sensor(
            depth, DEPTH_NOISE_RATE, noise_value=0.0)
    # return Image.fromarray(depth)
    return depth

def add_naive_noise_to_sensor(sensor_reading, noise_rate, noise_value=1.0):
        """
        Add naive sensor dropout to perceptual sensor, such as RGBD and LiDAR scan
        :param sensor_reading: raw sensor reading, range must be between [0.0, 1.0]
        :param noise_rate: how much noise to inject, 0.05 means 5% of the data will be replaced with noise_value
        :param noise_value: noise_value to overwrite raw sensor reading
        :return: sensor reading corrupted with noise
        """
        if noise_rate <= 0.0:
            return sensor_reading
        assert len(sensor_reading[(sensor_reading < 0.0) | (sensor_reading > 1.0)]) == 0,\
            'sensor reading has to be between [0.0, 1.0]'
        valid_mask = np.random.choice(2, sensor_reading.shape, p=[
                                      noise_rate, 1.0 - noise_rate])
        sensor_reading[valid_mask == 0] = noise_value
        return sensor_reading

class iGibsonInteractionPretrain(Dataset):
    """iGibson Interaction Pretrain dataset."""

    def __init__(self, 
                 load_depth=True,
                 transform=None,
                 depth_transform=None,
                 train=True):
        self.asset_root = gibson2.ig_dataset_path
        self.imgs = []
        scenes_root = os.path.join(self.asset_root, 'scenes')
        for s in os.listdir(scenes_root):
            pretrain_dir = os.path.join(scenes_root, s, 'misc/interaction_pretrain')
            if not os.path.isdir(pretrain_dir):
                continue
            data_range = range(3200) if train else range(3200, 4000)
            for i in data_range:
                for j in range(10):
                    img_path = os.path.join(pretrain_dir, 
                            '{:04d}'.format(i), 
                            'step_{:04d}_rgb.png'.format(j))
                    if os.path.isfile(img_path):
                        self.imgs.append(img_path)

        # self.imgs = glob.glob(os.path.join(self.asset_root, 'scenes',
                              # '*/misc/interaction_pretrain/*/step_000?_rgb.png'))
        if transform is None:
            normalize = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                            transforms.Resize(128),
                            transforms.ToTensor(),
                            normalize,])
        else:
            self.transform = transform
        self.load_depth = load_depth
        if load_depth:
            if depth_transform is None:
                self.depth_transform = transforms.Compose([
                                transforms.ToPILImage(),
                                transforms.Resize(128),
                                transforms.ToTensor(),])
            else:
                self.depth_transform = depth_transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        image = Image.open(img_name)
        image = self.transform(image)
        label, action = get_label(img_name)
        label = int(label)

        if self.load_depth:
            depth = get_depth_image(img_name)
            depth = self.depth_transform(depth)
            image = torch.cat((image, depth), 0)
        sample = {'image' : image, 
        # sample = {
                  'action': action, # (width, height)
                  'label' : label}
        return sample
