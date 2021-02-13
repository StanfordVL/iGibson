from gibson2.objects.pedestrian import Pedestrian
from gibson2.utils.utils import parse_config
from gibson2.utils.utils import create_directory
from gibson2.utils.utils import save_json_config
from gibson2.utils.utils import load_json_config
from gibson2.utils.utils import combine_paths
from gibson2.utils.utils import get_current_dir
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.episodes.episode_sample import EpisodeConfig
import argparse
import pybullet as p

import numpy as np
import collections


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
        env_config['use_sample_episode'] = False

    num_episodes = episode_config.get('num_episodes', 100)
    num_pedestrians = episode_config.get('num_pedestrians', 5)
    episode_length = episode_config.get(
        'episode_length', EpisodeConfig.MAX_EPISODE_LENGTH)
    numpy_seed = episode_config.get('numpy_seed', 1)
    orca_radius = env_config.get('orca_radius', 0.3)
    scene_id = env_config.get('scene_id', None)

    # Load the simulator
    env = iGibsonEnv(config_file=env_config,
                     mode='headless',
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 40.0)

    file_name = '{}_sample.json'.format(env.scene.scene_id)

    # simulator = load_simulator(env_config)
    # scene = load_scene(env_config, simulator)

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
        episode_info = episodeConfig.episodes[episode_index]

        for pedestrian_index in range(num_pedestrians):
            init_pos = episodeConfig.sample_initial_pos(
                env, pedestrian_index, episode_index)
            init_orientation = p.getQuaternionFromEuler(
                stubPedestrian.default_orn_euler)

            episode_info[pedestrian_index]['init_pos'] = init_pos
            episode_info[pedestrian_index]['init_orientation'] = init_orientation

            for _ in range(episode_length):
                _, target_pos = env.scene.get_random_point(
                    floor=env.task.floor_num)
                episode_info[pedestrian_index]['goal_pos'].append(target_pos)

    # Save episodeConfig object
    path = episodeConfig.save_scene_episodes(file_name)
    # load back episodeConfig object for sanity check
    # duplicateConfig = EpisodeConfig.load_scene_episode_config(path)
    # print(duplicateConfig)
    # print(episodeConfig)
    env.close()
