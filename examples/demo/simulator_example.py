import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
import pytest
import pybullet as p
import numpy as np
from gibson2.core.render.profiler import Profiler


def main():
    config = parse_config('../configs/turtlebot_demo.yaml')

    s = Simulator(mode='gui', image_width=512, image_height=512, render_to_tensor=False)
    scene = BuildingScene('Rs')
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    for i in range(1000):
        with Profiler('Simulator step'):
            turtlebot.apply_action([0.1,0.1])
            s.step()
            rgb = s.renderer.render_robot_cameras(modes=('rgb'))

    s.disconnect()


if __name__ == '__main__':
    main()
