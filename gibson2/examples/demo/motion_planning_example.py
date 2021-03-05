from gibson2.envs.igibson_env import iGibsonEnv
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
from gibson2.objects.ycb_object import YCBObject
import argparse
import numpy as np
from IPython import embed
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene

def run_example(args):
    nav_env = iGibsonEnv(config_file=args.config,
                                  mode=args.mode,
                                  action_timestep=1.0 / 120.0,
                                  physics_timestep=1.0 / 120.0)

    obstacles = []
    for _ in range(10):
        obj = YCBObject('003_cracker_box')
        nav_env.simulator.import_object(obj)
        obj.set_position_orientation(np.random.uniform(
            low=0, high=2, size=3), [0, 0, 0, 1])
        obstacles.append(obj.body_id)

    clutter_scene = InteractiveIndoorScene(
        "Rs_int", '{}_clutter'.format("Rs_int"))
    nav_env.simulator.import_non_colliding_objects(
        objects=clutter_scene.objects_by_name)

    motion_planner = MotionPlanningWrapper(nav_env)
    motion_planner.mp_obstacles += obstacles
    state = nav_env.reset()

    while True:
        action = np.zeros(nav_env.action_space.shape)
        state, reward, done, _ = nav_env.step(action)
        if motion_planner.attachment:
            motion_planner.attachment.assign()

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

