from gibson2.objects.pedestrian import Pedestrian
from gibson2.utils.utils import parse_config
from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.episodes.episode_sample import SocialNavEpisodesConfig
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
    episode_length = SocialNavEpisodesConfig.MAX_EPISODE_LENGTH
    raw_num_episodes = num_episodes * 3

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

            # Load the simulator
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 40.0)

            episode_config = SocialNavEpisodesConfig(
                num_pedestrians=env.task.num_pedestrians,
                scene_id=env.scene.scene_id,
                num_episodes=raw_num_episodes,
                orca_radius=env.task.orca_radius,
                numpy_seed=numpy_seed
            )

            # instance used to ccess pedestrian's default orientation
            for episode_index in range(raw_num_episodes):
                print('episode_index', episode_index)
                env.task.reset_agent(env)
                episode_info = episode_config.episodes[episode_index]
                episode_info['initial_pos'] = list(
                    env.robots[0].get_position())
                episode_info['initial_orn'] = list(
                    env.robots[0].get_orientation())
                episode_info['target_pos'] = list(env.task.target_pos)

                for i in range(env.task.num_pedestrians):
                    episode_info['pedestrians'][i]['initial_pos'] = \
                        list(env.task.pedestrians[i].get_position())
                    episode_info['pedestrians'][i]['initial_orn'] = \
                        list(env.task.pedestrians[i].get_orientation())
                    for step in range(episode_length):
                        _, target_pos = env.scene.get_random_point(
                            floor=env.task.floor_num)
                        episode_info['pedestrians'][i]['target_pos'][step] = \
                            list(target_pos)
            env.close()

            # Save episodeConfig object with 3x the episodes needed
            path = episode_config.save_scene_episodes(file_name)

            # Load these episodes one by one and filter in only episodes
            # in which the ORCA agent that represents the robot
            # reaches the goal withint the time limit
            env_config['load_scene_episode_config'] = True
            env_config['scene_episode_config_name'] = os.path.join(
                os.path.dirname(gibson2.__file__),
                'episodes', 'data', 'social_nav', file_name)
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 40.0)
            assert env.task.dist_tol > env.task.pedestrian_goal_thresh
            filtered_episodes = []
            for i in range(raw_num_episodes):
                env.task.reset_agent(env)
                shortest_path, _ = env.scene.get_shortest_path(
                    env.task.floor_num,
                    env.task.initial_pos[:2],
                    env.task.target_pos[:2],
                    entire_path=True)

                # Handle round off error during map_to_world/world_to_map conversion
                if np.linalg.norm(shortest_path[-1] - env.task.target_pos[:2]) > 0.01:
                    shortest_path = np.vstack(
                        [shortest_path, env.task.target_pos[:2]])
                waypoints = env.task.shortest_path_to_waypoints(shortest_path)

                for step in range(episode_length):
                    current_pos = env.robots[0].get_position()
                    next_goal = waypoints[0]
                    desired_vel = next_goal - current_pos[0:2]
                    desired_vel = desired_vel / \
                        np.linalg.norm(desired_vel) * \
                        env.task.orca_max_speed
                    env.task.orca_sim.setAgentPrefVelocity(
                        env.task.robot_orca_ped, tuple(desired_vel))
                    env.task.step(env)
                    pos = env.task.orca_sim.getAgentPosition(
                        env.task.robot_orca_ped)
                    env.robots[0].set_position(
                        [pos[0], pos[1], 0.0])

                    # Reached next weypoint
                    if np.linalg.norm(next_goal - np.array(pos)) <= env.task.pedestrian_goal_thresh:
                        waypoints.pop(0)

                    # Reached final goal
                    if np.linalg.norm(env.task.target_pos[:2] - np.array(pos)) <= env.task.dist_tol:
                        break

                success = (step != (episode_length - 1))
                print('episode', i, success)
                if success:
                    episode_config.episodes[i]['orca_timesteps'] = step
                    filtered_episodes.append(episode_config.episodes[i])
                    if len(filtered_episodes) == num_episodes:
                        break
            env.close()

            assert len(filtered_episodes) == num_episodes
            episode_config.episodes = filtered_episodes
            episode_config.num_episodes = num_episodes

            # Overwrite the original episode file
            path = episode_config.save_scene_episodes(file_name)
