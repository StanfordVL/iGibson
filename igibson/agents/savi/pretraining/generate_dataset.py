import os
import pickle
from itertools import product
import logging
import copy
import random

import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
from scipy.io import wavfile
from scipy.io.wavfile import write

import pybullet as p
from igibson.agents.savi.utils.dataset import CATEGORY_MAP
from igibson.agents.savi.utils.dataset import SCENE_SPLITS
from env_pretrain import AVNavRLEnv

src_dir = "/viscam/u/wangzz/avGibson/igibson/repo/iGibson-dev/igibson/agents/savi/"

def main(split):
    """
    split: 'train','val'
    """
    
    audios_ = list() 
    goals_ = list()
    
    source_sound_dir = f'/viscam/u/wangzz/avGibson/igibson/audio/semantic_splits/{split}'
    sound_files = os.listdir(source_sound_dir)
    sampling_rate = 44100
    
    for scene_name in tqdm(SCENE_SPLITS[split]):
        env = AVNavRLEnv(config_file=src_dir+'pretraining/config/pretraining.yaml', 
                                  mode='headless', scene_id=scene_name)
        goals = []
        for _ in tqdm(range(50)): # 50
            sound_file = random.choice(sound_files) # eg: sound_file = chair.wav
            index = CATEGORY_MAP[sound_file[:-4]] # remove .wav
            
            env.reset(split, sound_file)
            for _ in range(env.config["num_steps"]):
                action = env.action_space.sample()
                state, _, _, _ = env.step(action)

                spectro = state["audio"]
                audios_.append(spectro)

            if env.audio_system is not None:
                env.audio_system.reset()
                
            goal = np.zeros(env.config["num_steps"])
            goal[:] = index              
            goals.append(goal)

        goals_ += goals
        env.close()
        
    with open(os.path.join(src_dir + "pretraining/dataset/", '{}.pkl'.format(split)), 'wb') as fo:
        pickle.dump((audios_, goals_), fo)
        
        
if __name__ == '__main__':
    print('Caching train spectrograms ...')
    main("train")
#     print('Caching val spectrograms ...')
#     main("val")