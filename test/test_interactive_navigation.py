import yaml
from gibson2.core.physics.robot_locomotors import JR2_Kinova
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
from gibson2.core.physics.interactive_objects import InteractiveObj
import gibson2
import os
import numpy as np

config = parse_config('test_interactive_nav.yaml')


def test_jr2():
    s = Simulator(mode='gui')
    try:
        scene = BuildingScene('Ohoopee')
        s.import_scene(scene)
        jr2 = JR2_Kinova(config)
        s.import_robot(jr2)
        jr2.set_position([-6, 0, 0.1])
        obj3 = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components',
                                           'door.urdf'),
                              scale=2)
        s.import_interactive_object(obj3)
        obj3.set_position_rotation([-5, -1, 0], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])
        jr2.apply_action(np.array([0.005, 0.005, 0, 0, 0, 0, 0]))
        for _ in range(400):
            s.step()

        jr2.apply_action(np.array([0, 0, 0.002, 0.003, 0.001, 0.001, 0.002]))

        for _ in range(400):
            s.step()
    finally:
        s.disconnect()
