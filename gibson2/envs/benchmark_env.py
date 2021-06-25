import argparse
from gibson2.envs.behavior_env import BehaviorEnv
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default='gibson2/examples/configs/behavior.yaml',
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui', 'pbgui'],
                        default='gui',
                        help='which mode for simulation (default: headless)')
    parser.add_argument('--action_filter',
                        '-af',
                        choices=['navigation', 'tabletop_manipulation',
                                 'magic_grasping', 'mobile_manipulation',
                                 'all'],
                        default='mobile_manipulation',
                        help='which action filter')
    args = parser.parse_args()

    env = BehaviorEnv(config_file=args.config,
                      mode=args.mode,
                      action_timestep=1.0 / 30.0,
                      physics_timestep=1.0 / 300.0,
                      action_filter=args.action_filter)
    step_time_list = []
    start = time.time()
    env.reset()
    infos = []
    for i in range(1000):  # 10 seconds
        action = env.action_space.sample()
        action[:] = 0
        start = time.time()
        state, reward, done, info = env.step(action)
        elapsed = time.time()-start
        print(elapsed)
        print(info)
        if i > 900:
            infos.append(info)

    keys = ['time_physics', 'time_render', 'time_non_physics', 'time_checking', 'time_all']
    res = {}
    for key in keys:
        res[key] = 0
    for info in infos:
        for key in keys:
            res[key] += info[key]
    for key in keys:
        res[key] = res[key] / 100

    print(res)
    print('fps', 1/res['time_all'])
    print('num objects', len(env.scene.objects_by_id))
    #print('Episode finished after {} timesteps, took {} seconds.'.format(
    #    env.current_step, time.time() - start))
    env.close()
