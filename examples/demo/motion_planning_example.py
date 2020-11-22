from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import argparse
import numpy as np

def run_example(args):
    nav_env = iGibsonEnv(config_file=args.config,
                                  mode=args.mode,
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)

    motion_planner = MotionPlanningWrapper(nav_env)
    state = nav_env.reset()

    while True:
        action = np.zeros(nav_env.action_space.shape)
        state, reward, done, _ = nav_env.step(action)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
        '-m',
        choices=['headless', 'gui', 'iggui'],
        default='headless',
        help='which mode for simulation (default: headless)')

    args = parser.parse_args()
    run_example(args)

