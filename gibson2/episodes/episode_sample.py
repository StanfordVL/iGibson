from gibson2.utils.utils import create_directory
from gibson2.utils.utils import save_json_config
from gibson2.utils.utils import load_json_config
from gibson2.utils.utils import combine_paths
from gibson2.utils.utils import get_current_dir

import numpy as np


class EpisodeConfig:
    """
    Object that holds information about the sampled episodes
    for a particular scene.
    """

    MAX_EPISODE_LENGTH = 500
    BASE_DIR = combine_paths(get_current_dir(), '..', 'episodes/data')

    def __init__(self, num_pedestrians, scene_id, num_episodes, orca_radius=0.5, scene='igibson', numpy_seed=1):
        """
        num_pedestrians     The number of pedestrians to sample for.
        scene_id            Name of the scene to sample the episodes for
        num_episodes        Number of episodes to sample for
        orca_radius         The minimum distance between any two pedestrian's
                            initial positions
        scene               Either iGibson or Gibson
        numpy_seed          Seed number so that we can generate deterministic samples
        """
        np.random.seed(numpy_seed)
        self.numpy_seed = numpy_seed
        self.num_pedestrians = num_pedestrians
        self.num_episodes = num_episodes
        self.orca_radius = orca_radius
        self.scene_id = scene_id
        self.scene = scene
        # inital pos, goal pos, orientation
        self.episodes = [
            [{} for _ in range(num_pedestrians)] for _ in range(num_episodes)]
        for episode_index in range(num_episodes):
            for pedestrian_index in range(num_pedestrians):
                self.episodes[episode_index][pedestrian_index] = {
                    'init_pos': None, 'goal_pos': [], 'init_orientation': None}

        # reset_episode() is called first before this index is used.
        # We will eventually use zero-indexing
        self.episode_index = -1
        self.goal_index = [0] * num_pedestrians

    def reset_episode(self):
        self.goal_index = [0] * self.num_pedestrians
        self.episode_index += 1
        if self.episode_index >= len(self.episodes):
            raise ValueError(
                "We have exhausted all {} episode samples".format(len(self.episodes)))

    @classmethod
    def load_scene_episode_config(cls, path):
        """
        Class FactoryMethod to load episode samples from a particular file.
        It parses the json file and instantiates an EpisodeConfig object.

        :param path: the path to the configuration file
        :return episode_config: EpisodeConfig instance
        """
        config = load_json_config(path)
        episode_config = EpisodeConfig(
            num_pedestrians=config['config']['num_pedestrians'],
            scene_id=config['config']['scene_id'],
            num_episodes=config['config']['num_episodes'],
            orca_radius=config['config']['orca_radius'],
            scene=config['config']['scene'],
            numpy_seed=config['config']['numpy_seed']
        )
        num_pedestrians = config['config']['num_pedestrians']
        num_episodes = config['config']['num_episodes']

        for i in range(num_episodes):
            for j in range(num_pedestrians):
                episode_config.episodes[i][j]['init_pos'] = np.asarray(
                    config['episodes'][i][j]['init_pos'])
                episode_config.episodes[i][j]['goal_pos'] = list(
                    map(np.asarray, config['episodes'][i][j]['goal_pos']))
                episode_config.episodes[i][j]['init_orientation'] = config[
                    'episodes'][i][j]['init_orientation']
        return episode_config

    def sample_initial_pos(self, env, pedestrian_index, episode_index):
        """
        Samples the initial position for a pedestrian for a paricular episode.

        :param env: environment instance
        :param pedestrian_index: index of the pedestrian_key associated with
                                 this pedestrian
        :param episode_index: index of the episode_index associated with
                                 this episode
        """
        initial_pos = None
        must_resample_pos = True
        episode_info = self.episodes[episode_index]
        # resample pedestrian's initial position
        while must_resample_pos:
            initial_pos = None
            _, initial_pos = env.scene.get_random_point(
                floor=env.task.floor_num)

            must_resample_pos = False
            for neighbor_index in range(pedestrian_index):
                neighbor_pos_xyz = episode_info[neighbor_index]['init_pos']
                dist = np.linalg.norm([neighbor_pos_xyz[0] - initial_pos[0],
                                       neighbor_pos_xyz[1] - initial_pos[1]])
                if dist < self.orca_radius:
                    must_resample_pos = True
                    break
        return initial_pos

    def save_scene_episodes(self, filename):
        """
        Saves the scene episode to a .json file at path ./data/{scene_id}/{filename}

        :param filename: file name
        :return path: the absolute path of where the path was stored.
                      To use load_scene_episode_config class method, we need
                      to pass in this absolute path.
        """
        if not filename.endswith('.json'):
            filename += '.json'
        dir_path = combine_paths(get_current_dir(), 'data', self.scene_id)

        # creates directory if it does not exist.
        create_directory(dir_path)
        path = combine_paths(dir_path, filename)
        save_dict = {}
        tmp = [
            [{} for _ in range(self.num_pedestrians)] for _ in range(self.num_episodes)]

        for i in range(self.num_episodes):
            for j in range(self.num_pedestrians):
                tmp[i][j]['init_pos'] = self.episodes[i][j]['init_pos'].tolist(
                )
                tmp[i][j]['goal_pos'] = list(
                    map(list, self.episodes[i][j]['goal_pos']))
                tmp[i][j]['init_orientation'] = self.episodes[i][j]['init_orientation']

        save_dict['config'] = {}
        save_dict['config']['num_pedestrians'] = self.num_pedestrians
        save_dict['config']['num_episodes'] = self.num_episodes
        save_dict['config']['numpy_seed'] = self.numpy_seed
        save_dict['config']['orca_radius'] = self.orca_radius
        save_dict['config']['scene_id'] = self.scene_id
        save_dict['config']['scene'] = self.scene
        save_dict['episodes'] = tmp
        save_json_config(save_dict, path)
        return path
