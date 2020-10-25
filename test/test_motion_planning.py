import gibson2
from gibson2.envs.locomotor_env import NavigationEnv
from time import time
import os
from gibson2.utils.assets_utils import download_assets, download_demo_data
import numpy as np

def test_navigate_env():
    print("Test env")
    download_assets()
    download_demo_data()
    config_filename = os.path.join(gibson2.root_path, '../test/test_house_occupancy_grid.yaml')
    
    nav_env = NavigationEnv(config_file=config_filename, mode='headless')
    nav_env.reset()


    action = nav_env.action_space.sample()
    ts = nav_env.step(action)
    assert np.sum(ts[0]['occupancy_grid'] == 0) > 0
    assert np.sum(ts[0]['occupancy_grid'] == 1) > 0
    
    nav_env.clean()
