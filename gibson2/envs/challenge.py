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
        total_reward = 0
        total_success = 0
        for i in range(10):
            state = self.nav_env.reset()
            for step in range(500):
                action = agent.act(state)
                state, reward, done, info = self.nav_env.step(action)
                total_reward += reward
                if done:
                    break
            total_success +=  info['success']
            print('episode done, total reward {}, total success {}'.format(total_reward, total_success))

        return total_reward