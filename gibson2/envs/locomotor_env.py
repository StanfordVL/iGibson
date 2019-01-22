from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
import gibson2
from gibson2.utils.utils import parse_config
from gibson2.envs.base_env import BaseEnv

# define navigation environments following Anderson, Peter, et al. "On evaluation of embodied navigation agents." arXiv preprint arXiv:1807.06757 (2018).
# https://arxiv.org/pdf/1807.06757.pdf

class NavigateEnv(BaseEnv):
    def __init__(self, config_file, mode='headless'):
        super(NavigateEnv, self).__init__(config_file, mode)
        if self.config['task'] == 'pointgoal':
            self.target_pos = self.config['target_pos']
            self.target_orn = self.config['target_orn']
            self.initial_pos = self.config['initial_pos']
            self.initial_orn = self.config['initial_orn']

    def step(self, action):
        self.robots[0].apply_action(action)
        self.simulator_step()
        state = self.robots[0].calc_state()
        reward = 0
        return state, reward

    def reset(self):
        self.robots[0].robot_specific_reset()
        self.robots[0].set_position(pos=self.initial_pos)
        state = self.robots[0].calc_state()
        reward = 0
        return state, reward

if __name__ == "__main__":
    config_filename = os.path.join(os.path.dirname(gibson2.__file__), '../test/test.yaml')
    nav_env = NavigateEnv(config_file=config_filename, mode='gui')
    for j in range(100):
        if j%10 == 0:
            nav_env.set_mode('gui')
        else:
            nav_env.set_mode('headless')
        nav_env.reset()
        for i in range(100):
            nav_env.step(2)