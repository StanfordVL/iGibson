import gibson2
import json
import os
import math
import numpy as np
import glob
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import imageio

def get_label(image_path):
    img_dir = os.path.dirname(image_path)
    img_fname = os.path.splitext(os.path.basename(image_path))[0]
    img_id = img_fname.split('_')[1]
    json_path = os.path.join(img_dir, 'info_{}.json'.format(img_id))
    with open(json_path, 'r') as fp:
        data = json.load(fp)
    moved = get_binary_label(data)
    x,y = data['interact_at']
    x = int(x / 4) # width 
    y = int(y / 4) # height
    return moved, (x,y) 

def get_binary_label(data):
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
            return math.abs(joint_post['pos'] - joint_pre['pos']) > 0.1
        if joint_pre['type'] == 1:
            return math.abs(joint_post['pos'] - joint_pre['pos']) > 0.05

    return link_pos_delta > 0.1

def get_depth_path(image_path):
    return image_path.replace('_rgb.png', '_3d.png')
            
class iGibsonInteractionPretrain(Dataset):
    """iGibson Interaction Pretrain dataset."""

    def __init__(self, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.asset_root = gibson2.ig_dataset_path
        self.imgs = glob.glob(os.path.join(self.asset_root, 'scenes',
                              '*/misc/interaction_pretrain/*/step_000?_rgb.png'))
        if transform is None:
            normalize = transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            self.transform = transforms.Compose([
                            transforms.Resize(128),
                            transforms.ToTensor(),
                            normalize,])),
        else:
            self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = imgs[idx]
        image = imageio.imread(img_name)
        image = self.transform(image)
        label, action = int(get_binary_label(img_name))

        sample = {'image' : image, 
                  'action': action, # (width, height)
                  'label' : label}

        return sample
