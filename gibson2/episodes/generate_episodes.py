from gibson2.objects.pedestrian import Pedestrian
from gibson2.utils.utils import parse_config
from gibson2.utils.utils import create_directory
from gibson2.utils.utils import save_json_config
from gibson2.utils.utils import load_json_config
from gibson2.utils.utils import combine_paths
from gibson2.utils.utils import get_current_dir
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.episodes.episode_sample import SocialNavEpisodesConfig
import argparse
import pybullet as p

import numpy as np
import time


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
        raise ValueError('You must specify the .yaml configuration file'
                         ' for the task environment')
    else:
        env_config = parse_config(env_config_path)
        env_config['load_scene_episode_config'] = False

    num_episodes = episode_config.get('num_episodes', 100)
    episode_length = episode_config.get(
        'episode_length', SocialNavEpisodesConfig.MAX_EPISODE_LENGTH)
    numpy_seed = episode_config.get('numpy_seed', 1)
    # Load the simulator
    env = iGibsonEnv(config_file=env_config,
                     mode='headless',
                     action_timestep=1.0 / 10.0,
                     physics_timestep=1.0 / 40.0)

    print(env.task.radius, "!!!!!There you go")

    file_name = '{}_sample.json'.format(env.scene.scene_id)

    episode_config = SocialNavEpisodesConfig(
        num_pedestrians=env.task.num_pedestrians,
        scene_id=env.scene.scene_id,
        num_episodes=num_episodes,
        orca_radius=env.task.radius,
        numpy_seed=numpy_seed
    )

    print(episode_config)

    # instance used to ccess pedestrian's default orientation
    stubPedestrian = Pedestrian()

    for episode_index in range(num_episodes):
        env.reset()
        episode_info = episode_config.episodes[episode_index]

        for pedestrian_index in range(env.task.num_pedestrians):
            init_pos = episode_config.sample_initial_pos(
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
    path = episode_config.save_scene_episodes(file_name)

    # load back episodeConfig object for sanity check
    # path = '/cvgl2/u/jangj8523/iGibson/gibson2/episodes/data/Pomaria_2_int/Pomaria_2_int_sample.json'
    # duplicateConfig = EpisodeConfig.load_scene_episode_config(path)
    # print(duplicateConfig)
    # print(episodeConfig)
    env.close()

    # env_config['load_scene_episode_config'] = False
    # env_config['scene_episode_config_name'] = file_name
    #
    # env = iGibsonEnv(config_file=env_config,
    #                  mode='headless',
    #                  action_timestep=1.0 / 10.0,
    #                  physics_timestep=1.0 / 40.0)
    #
    # done_episode_index = []
    # for episode in range(10):
    #     print('Episode: {}'.format(episode))
    #     start = time.time()
    #     env.reset()
    #     for _ in range(100):  # 10 seconds
    #         action = env.action_space.sample()
    #         state, reward, done, _ = env.step(action)
    #         # print('state', state)
    #         if done:
    #             done_episode_index.append((episode, env.current_step))
    #             break
    #     print('Episode finished after {} timesteps, took {} seconds.'.format(
    #         env.current_step, time.time() - start))
    #
    # print("Size of successful episodes: {}".format(len(done_episode_index)))
    # env.task.episode_config.episodes = [episode_config.episodes[i]
    #                            for i in done_episode_index[:100]]
    # env.task.episode_config.num_episodes = len(env.task.episode_config.episodes)
    # env.task.episode_config.save_scene_episodes(
    #     'default_robot_task_complete_episodes({}_{})'.format(env.scene.scene_id, env_config['task']))
    #
    # env.close()
