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
from gibson2.utils.assets_utils import download_data

class Demo(object):
    def __init__(self):
        download_data()

    def run_demo(self):
        config = parse_config(os.path.join(gibson2.assets_path, '../../examples/configs/turtlebot_p2p_nav.yaml'))

        s = Simulator(mode='gui', image_width=700, image_height=700)
        scene = BuildingScene('Rs')
        s.import_scene(scene)
        turtlebot = Turtlebot(config)
        s.import_robot(turtlebot)

        for i in range(1000):
            turtlebot.apply_action([0.1,0.5])
            s.step()

        s.disconnect()


if __name__ == "__main__":
    Demo().run_demo()
