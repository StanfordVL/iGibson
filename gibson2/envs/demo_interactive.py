import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
import pytest
import pybullet as p
import numpy as np
import os
import gibson2
from gibson2.utils.assets_utils import download_assets, download_demo_data
import time
import matplotlib.pyplot as plt

class DemoInteractive(object):
    def __init__(self):
        download_assets()
        download_demo_data()

    def run_demo(self, optimize):
        s = Simulator(mode='iggui', image_width=700, image_height=700, optimize_render=optimize, timestep=1/60.0)
        scene = BuildingScene('Rs_interactive', is_interactive=True)
        scene.sleep = optimize
        s.import_scene(scene)
        if optimize:
            s.renderer.optimize_vertex_and_texture()

        fps = []
        for i in range(5000):
            #turtlebot.apply_action([0.1,0.5])
            start = time.time()
            s.step()
            elapsed = time.time() - start
            print(1/elapsed)
            fps.append(1/elapsed)
        s.disconnect()
        return fps


if __name__ == "__main__":
    res1 = DemoInteractive().run_demo(True)
    res2 = DemoInteractive().run_demo(False)
    
    from IPython import embed
    embed()