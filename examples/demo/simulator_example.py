import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
import pytest
import pybullet as p
import numpy as np
from gibson2.core.render.profiler import Profiler

config = parse_config('../configs/turtlebot_p2p_nav.yaml')

s = Simulator(mode='gui', resolution=512, render_to_tensor=False)
scene = BuildingScene('Bolton')
s.import_scene(scene)
turtlebot = Turtlebot(config)
s.import_robot(turtlebot)

p.resetBasePositionAndOrientation(scene.ground_plane_mjcf[0], posObj=[0, 0, -10], ornObj=[0, 0, 0, 1])

for i in range(2000):
    with Profiler('simulator step'):
        turtlebot.apply_action([0.1,0.1])
        s.step()
        rgb = s.renderer.render_robot_cameras(modes=('rgb'))

s.disconnect()