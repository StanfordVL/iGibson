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
            for _ in range(2):
                sound_file = random.choice(sound_files) # eg: sound_file = chair.wav
                index = CATEGORY_MAP[sound_file[:-4]] # remove .wav
                
                binaural_audio = self.compute_audio(sound_file) # change audio system in reset
                spectro = self.compute_spectrogram(binaural_audio)
                spectro = to_tensor(spectro)
                self.files.append(spectro)
               
                goal = to_tensor(np.zeros(1))
                goal[0] = index              
                goals.append(goal)

            self.goals += goals
            
            print(len(self.files), len(self.goals))
            self.env.close()

    def __len__(self):
        return len(self.goals)

    def __getitem__(self, item):
        inputs_outpus = (self.files[item], self.goals[item])
        return inputs_outputs 
    
    
    def compute_audio(self, sound_file):
        self.env.reset(self.split, sound_file)
        action = self.env.action_space.sample()
        for i in range(10):
            self.env.step(action)
        binaural_audio = self.env.complete_audio_output
        binaural_audio = np.array(binaural_audio)
        self.env.complete_audio_output = []
        
        if self.env.audio_system is not None:
            self.env.audio_system.disconnect()
            del self.env.audio_system
            self.env.audio_system = None
        return binaural_audio
    
    @staticmethod
    def compute_spectrogram(audiogoal):
        def compute_stft(signal):
            n_fft = 512
            hop_length = 160
            win_length = 400
            stft = np.abs(librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, win_length=win_length))
            stft = block_reduce(stft, block_size=(4, 4), func=np.mean)
            return stft
        audiogoal = audiogoal.astype(np.float32, order='C') / 32768.0
        channel1_magnitude = np.log1p(compute_stft(audiogoal[::2]))
        channel2_magnitude = np.log1p(compute_stft(audiogoal[1::2]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)
        return spectrogram