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
    return moved, (y,x) 

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
        self.asset_root = '/scr-ssd/wshen/dataset/ig_dataset'
        if not os.path.isdir(self.asset_root):
            self.asset_root = gibson2.ig_dataset_path
        self.imgs = []
        scenes_root = os.path.join(self.asset_root, 'scenes')
        for s in os.listdir(scenes_root):
            pretrain_dir = os.path.join(scenes_root, s, 'misc/interaction_pretrain')
            if not os.path.isdir(pretrain_dir):
                continue
            data_range = range(3600) if train else range(3600, 4000)
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
        x,y = action
        action = x * 128 + y
        label = int(label)

        if self.load_depth:
            depth = get_depth_image(img_name)
            depth = self.depth_transform(depth)
            image = torch.cat((image, depth), 0)
        sample = {'image' : image, 
        # sample = {
                  'action': np.array(action), # (height, width)
                  'label' : label}
        return sample

#######################
# Visualization 
#######################

from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def segmentation_pca( predicted ):
    H,W,C = predicted.shape
    from sklearn.decomposition import PCA  
    x = np.zeros((H,W,3), dtype='float')
    embedding_flattened = predicted.reshape((-1,C))
    pca = PCA(n_components=3)
    pca.fit(np.vstack(embedding_flattened))
    lower_dim = pca.transform(embedding_flattened).reshape((H,W,-1))
    x = (lower_dim - lower_dim.min()) / (lower_dim.max() - lower_dim.min())
    #scipy.misc.toimage(np.squeeze(x), cmin=0.0, cmax=1.0).save(to_store_name)
    return x

def visualize_data_entry(data_entry, feature, prediction, 
                         prediction_dense, 
                         step_num,
                         save_path=None,
                         save_first=True,
                         return_figure=False):
    for batch_i in range(data_entry['image'].size(0)):
        if data_entry['image'][batch_i].shape[0] == 4:
            depth = data_entry['image'][batch_i][-1,:,:].numpy()
            rgb_raw = data_entry['image'][batch_i][:-1,:,:]
        else:
            depth = np.zeros([128,128])
            rgb_raw = data_entry['image'][batch_i]
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
            std=[1/0.229, 1/0.224, 1/0.225]
        )
        im = transforms.ToPILImage()(inv_normalize(rgb_raw)).convert("RGB")
        draw = ImageDraw.Draw(im)
        xy_flat = data_entry['action'][batch_i]
        x = xy_flat // 128
        y = xy_flat % 128
        x_min = max(0, x-3)
        y_min = max(0, y-3)
        x_max = min(im.size[0], x+3)
        y_max = min(im.size[1], y+3)
        draw.ellipse((y_min, x_min, y_max, x_max), fill = 'blue', outline ='red')
        fig, ax = plt.subplots(nrows=2,ncols=2, figsize=(10,10))
        ax[0][0].imshow(im)
        ax[0][0].axis('off')
        ax[0][0].set_title('RGB')
        ax[0][1].imshow(depth)
        ax[0][1].axis('off')
        ax[0][1].set_title('Depth')

        feature_numpy = feature[batch_i].cpu().permute(1, 2, 0).numpy()
        feature_viz = segmentation_pca(feature_numpy)
        ax[1][1].imshow(feature_viz)
        ax[1][1].axis('off')
        ax[1][1].set_title('Feature')
        
        prediction_dense_np = prediction_dense[batch_i].cpu().permute(1,2,0).numpy()[:,:,1]
        pred_dense = plt.get_cmap('coolwarm')(prediction_dense_np)[:,:,:-1]
        #print(pred_dense.shape)
        pred_heatmap = Image.fromarray((pred_dense * 255).astype(np.uint8))
        overlayed = Image.blend(im, pred_heatmap, 0.5)
        #ax[1][0].imshow(im)
        ax[1][0].imshow(overlayed)
        ax[1][0].axis('off')
        ax[1][0].set_title('pred. (dense)')

        fig.suptitle('Interact {},{}; Label: {}; Prediction: {}'.format(x,y,
            'moved' if data_entry['label'][batch_i] else 'not moved',
            'moved' if np.argmax(prediction[batch_i].cpu().numpy()) 
            else 'not moved'), fontsize=20)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        if return_figure:
            return fig
        #plt.show()
        if save_path is not None:
            fig.savefig(os.path.join(save_path, 
                        '{:04d}_{:04d}.png'.format(step_num, batch_i)))
        fig.clf()
        plt.close()
        if save_first:
            break
