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
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.robots.turtlebot import Turtlebot
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.audio.audio_system import AudioSystem
from igibson.objects import cube
from igibson.utils.utils import (parse_config, l2_distance, quatToXYZW)
import pybullet as p
from igibson.external.pybullet_tools.utils import stable_z_on_aabb
from transforms3d.euler import euler2quat

from igibson.agents.savi.utils.dataset import CATEGORY_MAP
# from soundspaces.mp3d_utils import CATEGORY_INDEX_MAPPING


class AudioGoalDataset(Dataset):
    def __init__(self, scenes, split, use_polar_coordinates=False, use_cache=False, filter_rule='',
                target_dist_min = 1.0, target_dist_max = 6.0):
        self.split = split
        self.use_cache = use_cache
        self.files = list()
        self.goals = list()
        self.source_sound_dir = f'/viscam/u/wangzz/avGibson/igibson/audio/semantic_splits/{split}'
        self.source_sound_dict = dict()
        self.sampling_rate = 44100 # 16000
        sound_files = os.listdir(self.source_sound_dir)
        
        self.floor_num = 0
        self.target_dist_min = target_dist_min
        self.target_dist_max = target_dist_max       

        for scene in tqdm(scenes): # 9 scenes
            goals = []
            for _ in range(400):
                sound_file = random.choice(sound_files) # eg: sound_file = chair.wav
                index = CATEGORY_MAP[sound_file[:-4]] # remove .wav
                audiogoal = self.compute_audiogoal_from_scene(scene, sound_file, 
                                                              optimized=True, import_robot=True, num_sources=1)

                spectro = self.compute_spectrogram(audiogoal)
                self.files.append((scene, sound_file, spectro))

                goal = to_tensor(np.zeros(1))
                goal[0] = index               
                goals.append(goal)
            self.goals += goals

        self.data = [None] * len(self.goals)
        self.load_source_sounds()

    def audio_length(self, sound):
        return self.source_sound_dict[sound].shape[0] // self.sampling_rate

    def load_source_sounds(self):
        sound_files = os.listdir(self.source_sound_dir)
        for sound_file in sound_files:
            audio_data, sr = librosa.load(os.path.join(self.source_sound_dir, sound_file),
                                          sr=self.sampling_rate)
            self.source_sound_dict[sound_file] = audio_data


    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if (self.use_cache and self.data[item] is None) or not self.use_cache:
            scene, sound_file, spectrogram = self.files[item]     
            spectrogram = to_tensor(spectrogram)
            inputs_outputs = ([spectrogram], self.goals[item])

            if self.use_cache:
                self.data[item] = inputs_outputs
        else:
            inputs_outputs = self.data[item]

        return inputs_outputs
    
    
    def compute_audiogoal_from_scene(self, scene_name, sound_file, optimized, import_robot, num_sources):      
        config = parse_config('pretraining/config/pretraining_robot.yaml')
        scene = InteractiveIndoorScene(
            scene_name, texture_randomization=False, object_randomization=False)
        settings = MeshRendererSettings(
            msaa=False, enable_shadow=False)
        s = Simulator(mode='headless',
                      render_timestep=1 / 10.0,
                      physics_timestep=1 / 240.0,
                      image_width=128,
                      image_height=128,
                      device_idx=0,
                      rendering_settings=settings)
        s.import_ig_scene(scene)

        if import_robot:
            initial_pos, initial_orn, target_pos = self.sample_initial_pose_and_target_pos(scene)
            turtlebot = Turtlebot(config)
            s.import_robot(turtlebot)
            
            self.set_pos_orn_with_z_offset(turtlebot, initial_pos, initial_orn)
            turtlebot.robot_specific_reset()
            turtlebot.keep_still()
            body_id = turtlebot.robot_ids[0]
            land_success = False
            # land for maximum 1 second, should fall down ~5 meters
#             max_simulator_step = int(1.0 / self.action_timestep)
            max_simulator_step = int(1.0 / (1/10.0))
            for _ in range(max_simulator_step):
                s.step(audio=False)
                if len(p.getContactPoints(bodyA=body_id)) > 0:
                    land_success = True
                    break
            if not land_success:
                print("WARNING: Failed to land")
            turtlebot.robot_specific_reset()
            turtlebot.keep_still()
            
            if num_sources > 0:
                audioSystem = AudioSystem(s, turtlebot, is_Viewer=False, writeToFile=True, SR = 44100)
                for i in range(num_sources):
                    obj = cube.Cube(pos=target_pos, 
                                    dim=[0.05, 0.05, 0.05], 
                                    visual_only=False, mass=0.5, color=[255, 0, 0, 1])
                    obj_id = s.import_object(obj)
                    audioSystem.registerSource(obj_id, 
                                               "/viscam/u/wangzz/avGibson/igibson/audio/semantic_splits/"
                                               +self.split+"/"+sound_file, 
                                               enabled=True, repeat=False)
#                     audioSystem.setSourceRepeat(obj_id)
                s.attachAudioSystem(audioSystem)

        s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)

        for i in range(10): # sample 1s audio
            s.step()
        binaural_audio = s.audio_system.complete_output #
        s.disconnect()
        binaural_audio = np.array(binaural_audio)
        return binaural_audio

        
    def sample_initial_pose_and_target_pos(self, scene):
        """
        Sample robot initial pose and target position

        :param env: environment instance
        :return: initial pose and target position
        """
        _, initial_pos = scene.get_random_point(floor=self.floor_num)
        max_trials = 100
        dist = 0.0
        for _ in range(max_trials):
            _, target_pos = scene.get_random_point(floor=self.floor_num)
            if scene.build_graph:
                _, dist = scene.get_shortest_path(
                    self.floor_num,
                    initial_pos[:2],
                    target_pos[:2], entire_path=False)
            else:
                dist = l2_distance(initial_pos, target_pos)
            if self.target_dist_min < dist < self.target_dist_max:
                break
        if not (self.target_dist_min < dist < self.target_dist_max):
            print("WARNING: Failed to sample initial and target positions")
        initial_orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])
        return initial_pos, initial_orn, target_pos

    
    
    def set_pos_orn_with_z_offset(self, obj, pos, orn=None, offset=None):
        if orn is None:
            orn = np.array([0, 0, np.random.uniform(0, np.pi * 2)])

        if offset is None:
            offset = 0.1 #self.initial_pos_z_offset

        body_id = obj.robot_ids[0]
        # first set the correct orientation
        obj.set_position_orientation(pos, quatToXYZW(euler2quat(*orn), 'wxyz'))
        # compute stable z based on this orientation
        stable_z = stable_z_on_aabb(body_id, [pos, pos])
        # change the z-value of position with stable_z + additional offset
        # in case the surface is not perfect smooth (has bumps)
        obj.set_position([pos[0], pos[1], stable_z + offset])


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