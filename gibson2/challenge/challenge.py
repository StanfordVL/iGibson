from gibson2.utils.utils import parse_config
import numpy as np
import json
import os
from gibson2.envs.igibson_env import iGibsonEnv
import logging
logging.getLogger().setLevel(logging.WARNING)


class Challenge:
    def __init__(self):
        self.config_file = os.environ['CONFIG_FILE']
        self.split = os.environ['SPLIT']
        self.episode_dir = os.environ['EPISODE_DIR']
        self.eval_episodes_per_scene = os.environ.get(
            'EVAL_EPISODES_PER_SCENE', 100)

    def submit(self, agent):
        env_config = parse_config(self.config_file)

        task = env_config['task']
        if task == 'interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins', 'episode_return']}
        elif task == 'social_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'stl', 'psc', 'episode_return']}
        else:
            assert False, 'unknown task: {}'.format(task)

        num_episodes_per_scene = self.eval_episodes_per_scene
        split_dir = os.path.join(self.episode_dir, self.split)
        assert os.path.isdir(split_dir)
        num_scenes = len(os.listdir(split_dir))
        assert num_scenes > 0
        total_num_episodes = num_scenes * num_episodes_per_scene

        idx = 0
        for json_file in os.listdir(split_dir):
            scene_id = json_file.split('.')[0]
            json_file = os.path.join(split_dir, json_file)

            env_config['scene_id'] = scene_id
            env_config['load_scene_episode_config'] = True
            env_config['scene_episode_config_name'] = json_file
            env = iGibsonEnv(config_file=env_config,
                             mode='headless',
                             action_timestep=1.0 / 10.0,
                             physics_timestep=1.0 / 40.0)

            for _ in range(num_episodes_per_scene):
                idx += 1
                print('Episode: {}/{}'.format(idx, total_num_episodes))
                try:
                    agent.reset()
                except:
                    pass
                state = env.reset()
                episode_return = 0.0
                while True:
                    action = env.action_space.sample()
                    action = agent.act(state)
                    state, reward, done, info = env.step(action)
                    episode_return += reward
                    if done:
                        break

                metrics['episode_return'] += episode_return
                for key in metrics:
                    if key in info:
                        metrics[key] += info[key]

        for key in metrics:
            metrics[key] /= total_num_episodes
            print('Avg {}: {}'.format(key, metrics[key]))


if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
