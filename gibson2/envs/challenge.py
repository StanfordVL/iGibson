from gibson2.envs.locomotor_env import NavigateRandomEnvSim2Real
import os

class Challenge:
    def __init__(self):
        config_file = os.environ['CONFIG_FILE']
        sim2real_track = os.environ['SIM2REAL_TRACK']
        self.nav_env = NavigateRandomEnvSim2Real(config_file=config_file,
                                            mode='headless',
                                            action_timestep=1.0 / 10.0,
                                            physics_timestep=1.0 / 40.0,
                                            track=sim2real_track)

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
        print('eval done, avg reward {}, avg success {}, avg spl {}'.format(avg_reward, avg_success, avg_spl))
        return total_reward
