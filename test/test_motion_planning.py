import gibson2
from gibson2.envs.igibson_env import iGibsonEnv
from time import time
import os
from gibson2.utils.assets_utils import download_assets, download_demo_data
from gibson2.utils.motion_planning_wrapper import MotionPlanningWrapper
import numpy as np


def test_occupancy_grid():
    print("Test env")
    download_assets()
    download_demo_data()
    config_filename = os.path.join(gibson2.root_path, '../test/test_house_occupancy_grid.yaml')
    
    nav_env = iGibsonEnv(config_file=config_filename, mode='headless')
    nav_env.reset()

    action = nav_env.action_space.sample()
    ts = nav_env.step(action)
    assert np.sum(ts[0]['occupancy_grid'] == 0) > 0
    assert np.sum(ts[0]['occupancy_grid'] == 1) > 0

    nav_env.clean()


# def test_base_planning():
#     print("Test env")
#     download_assets()
#     download_demo_data()
#     config_filename = os.path.join(gibson2.root_path, '../test/test_house_occupancy_grid.yaml')

#     nav_env = NavigationEnv(config_file=config_filename, mode='headless')
#     motion_planner = MotionPlanningWrapper(nav_env)

#     state = nav_env.reset()
#     plan = motion_planner.plan_base_motion([0,1,0])
#     print(plan)
#     motion_planner.dry_run_base_plan(plan)

#     assert len(plan) > 0 
#     nav_env.clean()

