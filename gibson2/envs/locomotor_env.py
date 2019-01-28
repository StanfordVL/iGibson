from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
import gibson2
from gibson2.utils.utils import parse_config
from gibson2.envs.base_env import BaseEnv

# define navigation environments following Anderson, Peter, et al. "On evaluation of embodied navigation agents." arXiv preprint arXiv:1807.06757 (2018).
# https://arxiv.org/pdf/1807.06757.pdf

class NavigateEnv(BaseEnv):
    def __init__(self, config_file, mode='headless', action_timestep = 1/10.0, physics_timestep=1/240.0):
        super(NavigateEnv, self).__init__(config_file, mode)
        if self.config['task'] == 'pointgoal':
            self.target_pos = np.array(self.config['target_pos'])
            self.target_orn = np.array(self.config['target_orn'])
            self.initial_pos = np.array(self.config['initial_pos'])
            self.initial_orn = np.array(self.config['initial_orn'])
            self.potential = 1

        self.action_timestep = action_timestep
        self.physics_timestep = physics_timestep
        self.simulator.set_timestep(physics_timestep)
        self.simulator_loop = int(self.action_timestep / self.simulator.timestep)

        self.sensor_dim = self.robots[0].sensor_dim + 3
        self.action_dim = self.robots[0].action_dim

        obs_high = np.inf * np.ones(self.sensor_dim)
        self.observation_space = gym.spaces.Box(-obs_high, obs_high)
        self.action_space = self.robots[0].action_space
        self.current_step = 0
        self.max_step = 200

    def step(self, action):
        self.robots[0].apply_action(action)
        for i in range(self.simulator_loop):
            self.simulator_step()
        state = self.robots[0].calc_state()
        additional_state = self.target_pos - self.robots[0].get_position()
        state = np.concatenate((state, additional_state), 0)
        new_potential = np.sum((self.robots[0].get_position() - self.target_pos) ** 2) / np.sum((self.initial_pos - self.target_pos) ** 2)
        reward = self.potential - new_potential
        self.potential = new_potential

        self.current_step += 1
        done = self.current_step >= self.max_step

        return state, reward, done, {}

    def reset(self):
        self.robots[0].robot_specific_reset()
        self.robots[0].set_position(pos=self.initial_pos)
        state = self.robots[0].calc_state()
        additional_state = self.target_pos - self.robots[0].get_position()
        state = np.concatenate((state, additional_state), 0)
        self.current_step = 0
        return state


if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='gui')
    for j in range(15):
        if j%10 == 0:
            nav_env.set_mode('gui')
        else:
            nav_env.set_mode('headless')
        nav_env.reset()
        for i in range(300): # 300 steps, 30s world time
            action = nav_env.action_space.sample()
            ts = nav_env.step(action)
            print(ts)
            if ts[2]:
                print("Episode finished after {} timesteps".format(i + 1))
                break