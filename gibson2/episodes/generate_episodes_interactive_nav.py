from gibson2.objects.pedestrian import Pedestrian
from gibson2.utils.utils import parse_config
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.episodes.episode_sample import InteractiveNavEpisodesConfig
import argparse
import pybullet as p
import os
import gibson2

import numpy as np
import time
from IPython import embed


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
        help='which config file to use [default: use yaml files in /configs]',
        required=True)

    args = parser.parse_args()
    episode_config = parse_config(args.config)

    env_config_path = episode_config.get('env_config', None)
    if env_config_path is None:
        raise ValueError('You must specify the .yaml configuration file'
                         ' for the task environment')
    env_config = parse_config(env_config_path)
    num_episodes = episode_config.get('num_episodes', 100)
    numpy_seed = episode_config.get('numpy_seed', 0)

    dataset_split = {
        'minival': ['Rs_int'],
        'train': ['Merom_0_int', 'Benevolence_0_int', 'Pomaria_0_int',
                  'Wainscott_1_int', 'Rs_int', 'Ihlen_0_int',
                  'Beechwood_1_int', 'Ihlen_1_int'],
        'dev':  ['Benevolence_1_int', 'Wainscott_0_int'],
        'test': ['Pomaria_2_int', 'Benevolence_2_int', 'Beechwood_0_int',
                 'Pomaria_1_int', 'Merom_1_int']
    }

    for split in dataset_split:
        for scene_id in dataset_split[split]:
            file_name = os.path.join(split, '{}.json'.format(scene_id))

            env_config['load_scene_episode_config'] = False
            env_config['scene_episode_config_name'] = None
            env_config['scene_id'] = scene_id
            if split in ['dev', 'test']:
                env_config['use_test_objs'] = True
            else:
                env_config['use_test_objs'] = False

            # Load the simulator
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 240.0)

            episode_config = InteractiveNavEpisodesConfig(
                scene_id=env.scene.scene_id,
                num_episodes=num_episodes,
                numpy_seed=numpy_seed
            )

            # instance used to ccess pedestrian's default orientation
            for episode_index in range(num_episodes):
                print('episode_index', episode_index)
                env.task.reset_agent(env)
                episode_info = episode_config.episodes[episode_index]
                episode_info['initial_pos'] = list(
                    env.robots[0].get_position())
                episode_info['initial_orn'] = list(
                    env.robots[0].get_orientation())
                episode_info['target_pos'] = list(env.task.target_pos)
                episode_info['interactive_objects_idx'] = \
                    env.task.interactive_objects_idx.tolist()
                for i, obj in enumerate(env.task.interactive_objects):
                    episode_info['interactive_objects'].append({
                        'initial_pos': list(obj.get_position()),
                        'initial_orn': list(obj.get_orientation()),
                    })
            env.close()

            # Save episodeConfig object
            path = episode_config.save_scene_episodes(file_name)
