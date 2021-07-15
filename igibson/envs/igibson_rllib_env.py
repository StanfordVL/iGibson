"""
Example showing how to wrap the iGibson class using ray for rllib.
Multiple environments are only supported on Linux. If issues arise, please ensure torch/numpy
are installed *without* MKL support.

This example requires ray to be installed with rllib support, and pytorch to be installed:
    `pip install torch "ray[rllib]"`

Note: rllib only supports a single observation modality:
"""
import argparse

from gibson2.envs.igibson_env import iGibsonEnv

import ray
from ray.rllib.agents import ppo

ray.init()

class iGibsonRayEnv(iGibsonEnv):
    def __init__(self, env_config):
        super().__init__(
                config_file=env_config['config_file'],
                mode=env_config['mode'],
                action_timestep=env_config['action_timestep'],
                physics_timestep=env_config['physics_timestep'],
            )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default='gibson2/examples/configs/turtlebot_point_nav_ray.yaml',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()
    config = {
        "config_file": args.config,
        "mode": args.mode,
        "action_timestep": 1.0 / 10.0,
        "physics_timestep": 1.0 / 40.0
    }
    trainer = ppo.PPOTrainer(env=iGibsonRayEnv, config= {"env_config": config, "num_workers": 8, "framework": "torch"} )

    while True:
        print(trainer.train())

