from gibson2.envs.locomotor_env import NavigationRandomEnvSim2Real
import os
import json
import numpy as np


class Challenge:
    def __init__(self):
        self.config_file = os.environ['CONFIG_FILE']
        self.sim2real_track = os.environ['SIM2REAL_TRACK']
        self.nav_env = NavigationRandomEnvSim2Real(config_file=self.config_file,
                                                   mode='headless',
                                                   action_timestep=1.0 / 10.0,
                                                   physics_timestep=1.0 / 40.0,
                                                   track=self.sim2real_track)

    def submit(self, agent):
        total_reward = 0.0
        total_success = 0.0
        total_spl = 0.0
        num_eval_episodes = 10
        for i in range(num_eval_episodes):
            print('Episode: {}/{}'.format(i + 1, num_eval_episodes))
            state = self.nav_env.reset()
            while True:
                action = agent.act(state)
                state, reward, done, info = self.nav_env.step(action)
                total_reward += reward
                if done:
                    break
            total_success += info['success']
            total_spl += info['spl']

        avg_reward = total_reward / num_eval_episodes
        avg_success = total_success / num_eval_episodes
        avg_spl = total_spl / num_eval_episodes
        results = {}
        results["track"] = self.sim2real_track
        results["avg_spl"] = avg_spl
        results["avg_success"] = avg_success

        if os.path.exists('/results'):
            with open('/results/eval_result_{}.json'.format(self.sim2real_track), 'w') as f:
                json.dump(results, f)

        print('eval done, avg reward {}, avg success {}, avg spl {}'.format(avg_reward, avg_success, avg_spl))
        return total_reward

    def gen_episode(self):
        episodes = []
        for i in range(10):
            self.nav_env.reset()

            episode_info = {}
            episode_info['episode_id'] = str(i)
            episode_info['scene_id'] = self.nav_env.config['scene_id']
            episode_info['start_pos'] = list(self.nav_env.initial_pos.astype(np.float32))
            episode_info['end_pos'] = list(self.nav_env.target_pos.astype(np.float32))
            episode_info['start_rotation'] = list(self.nav_env.initial_orn.astype(np.float32))
            episode_info['end_rotation'] = list(self.nav_env.target_orn.astype(np.float32))
            episodes.append(episode_info)

        #with open('eval_episodes.json', 'w') as f:
        #    json.dump(str(episodes), f)

if __name__ == "__main__":
    challenge = Challenge()
    challenge.gen_episode()