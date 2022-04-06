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


class AudioGoalDataset(Dataset):
    def __init__(self, scenes, split):
        # scenes = SCENE_SPLITS[split]
        # split = "train" or "eval"
        self.split = split
        self.files = list()
        self.goals = list()
        self.source_sound_dir = f'/viscam/u/wangzz/avGibson/igibson/audio/semantic_splits/{split}'
        self.source_sound_dict = dict()
        self.sampling_rate = 44100 # 16000
        sound_files = os.listdir(self.source_sound_dir)
        
        self.floor_num = 0   
        for scene_name in tqdm(scenes): # 9 scenes
            self.env = AVNavRLEnv(config_file='pretraining/config/pretraining.yaml', 
                                  mode='headless', scene_id=scene_name) # write_to_file = True
            goals = []
            for _ in range(4000): # 20000
                sound_file = random.choice(sound_files) # eg: sound_file = chair.wav
                index = CATEGORY_MAP[sound_file[:-4]] # remove .wav
                
                self.compute_audio(sound_file) # change audio system in reset
               
                goal = to_tensor(np.zeros(self.env.config["num_steps"]))
                goal[:] = index              
                goals.append(goal)

            self.goals += goals
            
#             print(len(self.files), len(self.goals))
            self.env.close() # destroy audio system

    def __len__(self):
        return len(self.goals)

    def __getitem__(self, item):
        inputs_outputs = (self.files[item], self.goals[item])
        return inputs_outputs 
       
    def compute_audio(self, sound_file):
        self.env.reset(self.split, sound_file)
        for _ in range(self.env.config["num_steps"]):
            action = self.env.action_space.sample()
            state, _, _, _ = self.env.step(action)
            spectro = to_tensor(state["audio"])
            self.files.append(spectro)

        if self.env.audio_system is not None:
            self.env.audio_system.reset()
            