import os
import pickle
from itertools import product
import logging
import copy
import random

import librosa
import numpy as np
from torch.utils.data import Dataset
import networkx as nx
from tqdm import tqdm
from scipy.io import wavfile
from scipy.signal import fftconvolve
from skimage.measure import block_reduce

from igibson.agents.av_nav.utils.utils import to_tensor
from igibson.simulator import Simulator
from igibson.utils.utils import (parse_config, l2_distance, quatToXYZW)
import pybullet as p
from igibson.external.pybullet_tools.utils import stable_z_on_aabb
from transforms3d.euler import euler2quat

from igibson.agents.savi.utils.dataset import CATEGORY_MAP
from env_pretrain import AVNavRLEnv
# from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING

src_dir = "/viscam/u/wangzz/avGibson/igibson/repo/iGibson-dev/igibson/agents/savi/"

class AudioGoalDataset(Dataset):
    def __init__(self, scenes, split):
        # scenes = SCENE_SPLITS[split]
        # split = "train" or "eval"
        self.split = split
        with open(os.path.join(src_dir + "pretraining/dataset/", '{}.pkl'.format(split)), 'rb') as fi:
            audios, goals = pickle.load(fi)       
        self.files = audios
        self.goals = np.array(goals).flatten().astype('int').tolist()

    def __len__(self):
        return len(self.goals)

    def __getitem__(self, item):
        inputs_outputs = (self.files[item], self.goals[item])
        return inputs_outputs 
       

