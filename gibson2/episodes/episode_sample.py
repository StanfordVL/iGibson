import gibson2
import numpy as np
import os
import json


class EpisodeConfig:
    def __init__(self, scene_id, num_episodes, numpy_seed=0):
        np.random.seed(numpy_seed)
        self.scene_id = scene_id
        self.num_episodes = num_episodes
        self.numpy_seed = numpy_seed
        # reset_episode() is called first before this index is used.
        # We will eventually use zero-indexing
        self.episode_index = -1
        self.episodes = [{} for _ in range(num_episodes)]

    def reset_episode(self):
        self.episode_index += 1
        if self.episode_index >= self.num_episodes:
            raise ValueError(
                "We have exhausted all {} episode samples".format(
                    self.num_episodes))


class SocialNavEpisodesConfig(EpisodeConfig):
    """
    Object that holds information about the sampled episodes
    for a particular scene.
    """
    MAX_EPISODE_LENGTH = 500

    def __init__(self, scene_id, num_episodes, num_pedestrians, orca_radius, numpy_seed=0):
        """
        scene_id            Name of the scene to sample the episodes for
        num_episodes        Number of episodes to sample for
        num_pedestrians     The number of pedestrians to sample for.
        orca_radius         The minimum distance between any two pedestrian's
                            initial positions
        numpy_seed          Seed number so that we can generate deterministic samples
        """
        super(SocialNavEpisodesConfig, self).__init__(
            scene_id, num_episodes, numpy_seed)
        self.num_pedestrians = num_pedestrians
        self.orca_radius = orca_radius

        # inital pos, goal pos, orientation
        for episode_index in range(num_episodes):
            self.episodes[episode_index] = {
                'initial_pos': None,
                'initial_orn': None,
                'target_pos': None,
                'orca_timesteps': None,
                'pedestrians': [],
            }
            for pedestrian_index in range(num_pedestrians):
                self.episodes[episode_index]['pedestrians'].append({
                    'initial_pos': None,
                    'target_pos': [None] * self.MAX_EPISODE_LENGTH,
                    'initial_orn': None})
        self.goal_index = [0] * num_pedestrians

    def reset_episode(self):
        super(SocialNavEpisodesConfig, self).reset_episode()
        self.goal_index = [0] * self.num_pedestrians

    @classmethod
    def load_scene_episode_config(cls, path):
        """
        Class FactoryMethod to load episode samples from a particular file.
        It parses the json file and instantiates an EpisodeConfig object.

        :param path: the path to the configuration file
        :return episode_config: EpisodeConfig instance
        """
        with open(path) as f:
            config = json.load(f)
        episode_config = SocialNavEpisodesConfig(
            scene_id=config['config']['scene_id'],
            num_episodes=config['config']['num_episodes'],
            num_pedestrians=config['config']['num_pedestrians'],
            orca_radius=config['config']['orca_radius'],
            numpy_seed=config['config']['numpy_seed']
        )
        episode_config.episodes = config['episodes']

        return episode_config

    def save_scene_episodes(self, filename):
        """
        Saves the scene episode to a .json file at path ./data/{scene_id}/{filename}

        :param filename: file name
        :return path: the absolute path of where the path was stored.
                      To use load_scene_episode_config class method, we need
                      to pass in this absolute path.
        """
        dir_path = os.path.join(
            os.path.dirname(gibson2.__file__),
            'episodes', 'data', 'social_nav')
        path = os.path.join(dir_path, filename)

        save_dict = {}
        save_dict['config'] = {}
        save_dict['config']['num_pedestrians'] = self.num_pedestrians
        save_dict['config']['num_episodes'] = self.num_episodes
        save_dict['config']['numpy_seed'] = self.numpy_seed
        save_dict['config']['orca_radius'] = self.orca_radius
        save_dict['config']['scene_id'] = self.scene_id
        save_dict['episodes'] = self.episodes
        with open(path, 'w+') as f:
            json.dump(save_dict, f, sort_keys=True, indent=4)
        return path


class InteractiveNavEpisodesConfig(EpisodeConfig):
    """
    Object that holds information about the sampled episodes
    for a particular scene.
    """

    def __init__(self, scene_id, num_episodes, numpy_seed=0):
        """
        scene_id            Name of the scene to sample the episodes for
        num_episodes        Number of episodes to sample for
        num_pedestrians     The number of pedestrians to sample for.
        orca_radius         The minimum distance between any two pedestrian's
                            initial positions
        numpy_seed          Seed number so that we can generate deterministic samples
        """
        super(InteractiveNavEpisodesConfig, self).__init__(
            scene_id, num_episodes, numpy_seed)

        # inital pos, goal pos, orientation
        for episode_index in range(num_episodes):
            self.episodes[episode_index] = {
                'initial_pos': None,
                'initial_orn': None,
                'target_pos': None,
                'interactive_objects': [],
                'interactive_objects_idx': [],
            }

    @classmethod
    def load_scene_episode_config(cls, path):
        """
        Class FactoryMethod to load episode samples from a particular file.
        It parses the json file and instantiates an EpisodeConfig object.

        :param path: the path to the configuration file
        :return episode_config: EpisodeConfig instance
        """
        with open(path) as f:
            config = json.load(f)
        episode_config = InteractiveNavEpisodesConfig(
            scene_id=config['config']['scene_id'],
            num_episodes=config['config']['num_episodes'],
            numpy_seed=config['config']['numpy_seed']
        )
        episode_config.episodes = config['episodes']

        return episode_config

    def save_scene_episodes(self, filename):
        """
        Saves the scene episode to a .json file at path ./data/{scene_id}/{filename}

        :param filename: file name
        :return path: the absolute path of where the path was stored.
                      To use load_scene_episode_config class method, we need
                      to pass in this absolute path.
        """
        dir_path = os.path.join(
            os.path.dirname(gibson2.__file__),
            'episodes', 'data', 'interactive_nav')

        path = os.path.join(dir_path, filename)

        save_dict = {}
        save_dict['config'] = {}
        save_dict['config']['num_episodes'] = self.num_episodes
        save_dict['config']['numpy_seed'] = self.numpy_seed
        save_dict['config']['scene_id'] = self.scene_id
        save_dict['episodes'] = self.episodes
        with open(path, 'w+') as f:
            json.dump(save_dict, f, sort_keys=True, indent=4)
        return path
