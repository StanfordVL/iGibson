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
from igibson.robots.turtlebot_robot import Turtlebot
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
#         self.binaural_rir_dir = 'data/binaural_rirs/mp3d'
        self.source_sound_dir = f'/viscam/u/wangzz/avGibson/igibson/audio/semantic_splits/{split}'
#         self.source_sound_dir = f'/viscam/u/wangzz/SoundSpaces/sound-spaces/data/sounds/semantic_splits/{split}'
        self.source_sound_dict = dict()
        self.sampling_rate = 44100 # 16000
        sound_files = os.listdir(self.source_sound_dir)
        
        self.floor_num = 0
        self.target_dist_min = target_dist_min
        self.target_dist_max = target_dist_max
        

        for scene in tqdm(scenes): # 8 scenes
#             scene_graph = scene_graphs[scene]
            goals = []
#             subgraphs = list(nx.connected_components(scene_graph))
#             #[{33, 2, 3, 4, 34, 5, 35, 68, 36, 69, 16, 17, 18, 53, 54}, {6, 39, 40, ...
#             sr_pairs = list() # source receiver pairs
#             for subgraph in subgraphs:
#                 sr_pairs += list(product(subgraph, subgraph))
#             random.shuffle(sr_pairs)
#             for s, r in sr_pairs[:50000]:
            for _ in range(400): # 20
                sound_file = random.choice(sound_files) # eg: sound_file = chair.wav
                index = CATEGORY_MAP[sound_file[:-4]] # remove .wav
                audiogoal = self.compute_audiogoal_from_scene(scene, sound_file, 
                                                              optimized=True, import_robot=True, num_sources=1)
#                 print("audiogoal", audiogoal)
                spectro = self.compute_spectrogram(audiogoal)
                self.files.append((scene, sound_file, spectro))
#                 angle = random.choice([0, 90, 180, 270])
#                 rir_file = os.path.join(self.binaural_rir_dir, scene, str(angle), f"{r}_{s}.wav")
#                 self.files.append((rir_file, sound_file))
                
#                 delta_x = scene_graph.nodes[s]['point'][0] - scene_graph.nodes[r]['point'][0]
#                 delta_y = scene_graph.nodes[s]['point'][2] - scene_graph.nodes[r]['point'][2]
#                 goal_xy = self._compute_goal_xy(delta_x, delta_y, angle, use_polar_coordinates)

#                 goal = to_tensor(np.zeros(3))
                
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

#     @staticmethod
#     def _compute_goal_xy(delta_x, delta_y, angle, use_polar_coordinates):
#         """
#         -Y is forward, X is rightward, agent faces -Y
#         """
#         if angle == 0:
#             x = delta_x
#             y = delta_y
#         elif angle == 90:
#             x = delta_y
#             y = -delta_x
#         elif angle == 180:
#             x = -delta_x
#             y = -delta_y
#         else:
#             x = -delta_y
#             y = delta_x

#         if use_polar_coordinates:
#             theta = np.arctan2(y, x)
#             distance = np.linalg.norm([y, x])
#             goal_xy = to_tensor([theta, distance])
#         else:
#             goal_xy = to_tensor([x, y])
#         return goal_xy

    def __len__(self):
        return len(self.files)

    def __getitem__(self, item):
        if (self.use_cache and self.data[item] is None) or not self.use_cache:
#             rir_file, sound_file = self.files[item]
#             audiogoal = self.compute_audiogoal(rir_file, sound_file)
            # audiogoal: 16000*2
            
            scene, sound_file, spectrogram = self.files[item]
#             audiogoal = self.compute_audiogoal_from_scene(scene, sound_file, optimized=True, import_robot=True, num_sources=1)
            
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
            msaa=False, enable_shadow=False) #, optimized=optimized)
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
            
            # self.land_robot(turtlebot, initial_pos, initial_orn)
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
#                     _,source_location = scene.get_random_point_by_room_type("living_room")
#                     target_pos[2] = 1.7
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

        for i in range(10):
            s.step()
#         s.step()
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
        print("l2_distance", dist)
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

    
#     def compute_audiogoal(self, binaural_rir_file, sound_file):
#         sampling_rate = self.sampling_rate
#         try:
#             sampling_freq, binaural_rir = wavfile.read(binaural_rir_file)  # float32
#         except ValueError:
#             logging.warning("{} file is not readable".format(binaural_rir_file))
#             binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)
#         if len(binaural_rir) == 0:
#             logging.debug("Empty RIR file at {}".format(binaural_rir_file))
#             binaural_rir = np.zeros((sampling_rate, 2)).astype(np.float32)

#         current_source_sound = self.source_sound_dict[sound_file]
#         index = random.randint(0, self.audio_length(sound_file) - 2)
#         if index * sampling_rate - binaural_rir.shape[0] < 0:
#             source_sound = current_source_sound[: (index + 1) * sampling_rate]
#             binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel]
#                                                        ) for channel in range(binaural_rir.shape[-1])])
#             audiogoal = binaural_convolved[:, index * sampling_rate: (index + 1) * sampling_rate]
#         else:
#             # include reverb from previous time step
#             source_sound = current_source_sound[index * sampling_rate - binaural_rir.shape[0]
#                                                 : (index + 1) * sampling_rate]
#             binaural_convolved = np.array([fftconvolve(source_sound, binaural_rir[:, channel], mode='valid',
#                                                        ) for channel in range(binaural_rir.shape[-1])])
#             audiogoal = binaural_convolved[:, :-1]

#         return audiogoal

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
#         print("audiogoal[::2]", len(audiogoal[::2]))
        channel1_magnitude = np.log1p(compute_stft(audiogoal[::2]))
        channel2_magnitude = np.log1p(compute_stft(audiogoal[1::2]))
        spectrogram = np.stack([channel1_magnitude, channel2_magnitude], axis=-1)
#         print("spectrogram", spectrogram.shape)
        return spectrogram