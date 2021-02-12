from gibson2.objects.pedestrian import Pedestrian
from gibson2.utils.utils import parse_config
from gibson2.utils.utils import create_directory
from gibson2.utils.utils import save_json_config
from gibson2.utils.utils import load_json_config
from gibson2.utils.utils import combine_paths
from gibson2.utils.utils import get_current_dir
from gibson2.envs.igibson_env import iGibsonEnv
import argparse
import pybullet as p

# import gym
import numpy as np
import collections


class EpisodeConfig:
    """
    Object that holds information about the sampled episodes
    for a particular scene.
    """

    MAX_EPISODE_LENGTH = 500

    def __init__(self, num_pedestrians, scene_id, num_episodes, orca_radius=0.3, scene='igibson', numpy_seed=1):
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
        self.episode_keys = ['episode {}'.format(
            i) for i in range(num_episodes)]
        self.pedestrian_keys = ['pedestrian {}'.format(
            i) for i in range(num_pedestrians)]
        self.num_pedestrians = num_pedestrians
        self.num_episodes = num_episodes
        self.orca_radius = orca_radius
        self.scene_id = scene_id
        self.scene = scene
        # inital pos, goal pos, orientation
        self.episodes = {}
        for episode_key in self.episode_keys:
            self.episodes[episode_key] = {}
            for pedestrian_key in self.pedestrian_keys:
                self.episodes[episode_key][pedestrian_key] = {
                    'init_pos': None, 'goal_pos': [], 'init_orientation': None}

        self.episode_index = 0
        self.goal_index = [0] * num_pedestrians

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
            numpy_seed=numpy_seed
        )
        num_pedestrians = config['config']['num_pedestrians']
        num_episodes = config['config']['num_episodes']

        episode_keys = ['episode {}'.format(
            i) for i in range(num_episodes)]
        pedestrian_keys = ['pedestrian {}'.format(
            i) for i in range(num_pedestrians)]

        for episode_key in episode_keys:
            for pedestrian_key in pedestrian_keys:
                episode_config.episodes[episode_key][pedestrian_key]['init_pos'] = np.asarray(
                    config['episodes'][episode_key][pedestrian_key]['init_pos'])
                episode_config.episodes[episode_key][pedestrian_key]['goal_pos'] = list(
                    map(np.asarray, config['episodes'][episode_key][pedestrian_key]['goal_pos']))
                episode_config.episodes[episode_key][pedestrian_key]['init_orientation'] = config[
                    'episodes'][episode_key][pedestrian_key]['init_orientation']
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
        episode_key = self.episode_keys[episode_index]
        episode_info = self.episodes[episode_key]
        # resample pedestrian's initial position
        while must_resample_pos:
            initial_pos = None
            _, initial_pos = env.scene.get_random_point(
                floor=env.task.floor_num)

            must_resample_pos = False
            for i in range(pedestrian_index):
                neighbor_key = self.pedestrian_keys[i]
                neighbor_pos_xyz = episode_info[neighbor_key]['init_pos']
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

        for episode_key in self.episode_keys:
            for pedestrian_key in self.pedestrian_keys:
                self.episodes[episode_key][pedestrian_key]['init_pos'] = self.episodes[episode_key][pedestrian_key]['init_pos'].tolist()
                self.episodes[episode_key][pedestrian_key]['goal_pos'] = list(
                    map(list, self.episodes[episode_key][pedestrian_key]['goal_pos']))

        save_dict['config'] = {}
        save_dict['config']['num_pedestrians'] = self.num_pedestrians
        save_dict['config']['num_episodes'] = self.num_episodes
        save_dict['config']['numpy_seed'] = self.numpy_seed
        save_dict['config']['orca_radius'] = self.orca_radius
        save_dict['config']['scene_id'] = self.scene_id
        save_dict['config']['scene'] = self.scene
        save_dict['episodes'] = self.episodes
        save_json_config(save_dict, path)
        return path


if __name__ == '__main__':
    """
    Generates the sample episodes based on the config file provided by the
    user. The config file should contain the path to the .yaml that holds
    all the necessary information about the simulation.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in /configs]')

    args = parser.parse_args()
    episode_config = parse_config(args.config)

    env_config_path = episode_config.get('env_config', None)
    if env_config_path is None:
        raise ValueError('You must specify the .json files that you would use for'
                         ' for running a task experiment')
    else:
        env_config = parse_config(env_config_path)

    num_episodes = episode_config.get('num_episodes', 100)
    num_pedestrians = episode_config.get('num_pedestrians', 5)
    episode_length = episode_config.get(
        'episode_length', EpisodeConfig.MAX_EPISODE_LENGTH)
    numpy_seed = episode_config.get('numpy_seed', 1)
    orca_radius = env_config.get('orca_radius', 0.3)
    scene_id = env_config['scene_id']
    file_name = '{}_sample.json'.format(scene_id)

    env = iGibsonEnv(config_file=env_config_path,
                     mode='headless',
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 40.0)

    episodeConfig = EpisodeConfig(
        num_pedestrians=num_pedestrians,
        scene_id=scene_id,
        num_episodes=num_episodes,
        orca_radius=orca_radius,
        numpy_seed=numpy_seed
    )

    # hack to access pedestrian's default orientation
    stubPedestrian = Pedestrian()

    for episode_index in range(num_episodes):
        env.reset()
        episode_key = episodeConfig.episode_keys[episode_index]
        episode_info = episodeConfig.episodes[episode_key]

        for pedestrian_index in range(num_pedestrians):
            pedestrian_key = episodeConfig.pedestrian_keys[pedestrian_index]

            init_pos = episodeConfig.sample_initial_pos(
                env, pedestrian_index, episode_index)
            init_orientation = p.getQuaternionFromEuler(
                stubPedestrian.default_orn_euler)

            episode_info[pedestrian_key]['init_pos'] = init_pos
            episode_info[pedestrian_key]['init_orientation'] = init_orientation

            for _ in range(episode_length):
                _, target_pos = env.scene.get_random_point(
                    floor=env.task.floor_num)
                episode_info[pedestrian_key]['goal_pos'].append(target_pos)

    # Save episodeConfig object
    path = episodeConfig.save_scene_episodes(file_name)

    # load back episodeConfig object for sanity check
    # duplicateConfig = EpisodeConfig.load_scene_episode_config(path)
    # print(duplicateConfig)
    # print(episodeConfig)
    env.close()
