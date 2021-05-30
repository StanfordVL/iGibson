import argparse
import numpy as np
import time
import tasknet
import types
import gym.spaces
import pybullet as p

from collections import OrderedDict
from gibson2.robots.behavior_robot import BehaviorRobot
from gibson2.envs.behavior_env import BehaviorEnv

class BehaviorMPEnv(BehaviorEnv):
    """
    iGibson Environment (OpenAI Gym interface)
    """

    def __init__(
        self,
        config_file,
        scene_id=None,
        mode='headless',
        action_timestep=1 / 10.0,
        physics_timestep=1 / 240.0,
        device_idx=0,
        render_to_tensor=False,
        automatic_reset=False,
        seed=0,
        action_filter='mobile_manipulation'
    ):
        """
        :param config_file: config_file path
        :param scene_id: override scene_id in config file
        :param mode: headless, gui, iggui
        :param action_timestep: environment executes action per action_timestep second
        :param physics_timestep: physics timestep for pybullet
        :param device_idx: which GPU to run the simulation and rendering on
        :param render_to_tensor: whether to render directly to pytorch tensors
        :param automatic_reset: whether to automatic reset after an episode finishes
        """
        super(BehaviorMPEnv, self).__init__(config_file=config_file,
                                            scene_id=scene_id,
                                            mode=mode,
                                            action_timestep=action_timestep,
                                            physics_timestep=physics_timestep,
                                            device_idx=device_idx,
                                            render_to_tensor=render_to_tensor,
                                            action_filter=action_filter,
                                            seed=seed,
                                            automatic_reset=automatic_reset)

        super(BehaviorMPEnv, self).reset()
        super(BehaviorMPEnv, self).step(np.zeros(17))
        super(BehaviorMPEnv, self).step(np.zeros(17))

    def load_action_space(self):
        self.action_space = gym.spaces.Discrete(10)

    def step(self, action):

        # Do magic action or MP action here

        state, reward, done, info = super(BehaviorMPEnv, self).step(np.zeros(17))
        return state, reward, done, info

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default = 'gibson2/examples/configs/behavior.yaml',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui', 'pbgui'],
                        default='gui',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = BehaviorMPEnv(config_file=args.config,
                      mode=args.mode,
                      action_timestep=1.0 / 10.0,
                      physics_timestep=1.0 / 40.0)
    step_time_list = []
    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        env.reset()
        for i in range(1000):  # 10 seconds
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            if done:
                break
        print('Episode finished after {} timesteps, took {} seconds.'.format(
            env.current_step, time.time() - start))
    env.close()
