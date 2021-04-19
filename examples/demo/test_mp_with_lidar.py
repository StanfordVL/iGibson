from gibson2.envs.motion_planner_env import MotionPlanningBaseArmEnv
import argparse, time


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui'],
                        default='headless',
                        help='which mode for simulation (default: headless)')

    args = parser.parse_args()

    nav_env = MotionPlanningBaseArmEnv(config_file=args.config,
                                       mode=args.mode,
                                       action_timestep=1.0 / 1000000.0,
                                       physics_timestep=1.0 / 1000000.0,
                                        eval=True)

    for episode in range(100):
        print('Episode: {}'.format(episode))
        start = time.time()
        state = nav_env.reset()
        for i in range(150):

            from IPython import embed; embed()
            print(state['pc'])


            # action = nav_env.action_space.sample()
            # state, reward, done, info = nav_env.step(action)

            # if done:
            #    print('Episode finished after {} timesteps'.format(i + 1))
            #    break
        print(time.time() - start)
    nav_env.clean()
