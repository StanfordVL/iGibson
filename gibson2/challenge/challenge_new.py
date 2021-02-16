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
        self.phase = os.environ['PHASE']
        self.episode_dir = os.environ['EPISODE_DIR']

    def submit(self, agent):
        env_config = parse_config(self.config_file)

        task = env_config['task']
        if task == 'interactive_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'spl', 'effort_efficiency', 'ins']}
        elif task == 'social_nav_random':
            metrics = {key: 0.0 for key in [
                'success', 'stl', 'psc']}
        else:
            assert False, 'unknown task: {}'.format(task)

        num_episodes_per_scene = 3
        phase_dir = os.path.join(self.episode_dir, self.phase)
        assert os.path.isdir(phase_dir)
        num_scenes = len(os.listdir(phase_dir))
        assert num_scenes > 0
        total_num_episodes = num_scenes * num_episodes_per_scene

        idx = 0
        for json_file in os.listdir(phase_dir):
            scene_id = json_file.split('.')[0]
            json_file = os.path.join(phase_dir, json_file)

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
                # try:
                #     agent.reset()
                # except:
                #     pass
                state = env.reset()
                while True:
                    action = env.action_space.sample()
                    # action = agent.act(state)
                    state, reward, done, info = env.step(action)
                    # total_reward += reward
                    if done:
                        break
                print(info)
                for key in metrics:
                    metrics[key] += info[key]

        for key in metrics:
            metrics[key] /= total_num_episodes
            print('avg {}: {}'.format(key, metrics[key]))

            # avg_reward = total_reward / num_eval_episodes
            # avg_success = total_success / num_eval_episodes
            # avg_spl = total_spl / num_eval_episodes
            # results = {}
            # results["track"] = self.track
            # results["avg_spl"] = avg_spl
            # results["avg_success"] = avg_success

            # if os.path.exists('/results'):
            #     with open('/results/eval_result_{}.json'.format(self.track), 'w') as f:
            #         json.dump(results, f)

            # print('eval done, avg reward {}, avg success {}, avg spl {}'.format(
            #     avg_reward, avg_success, avg_spl))
            # return total_reward


if __name__ == '__main__':
    challenge = Challenge()
    challenge.submit(None)
