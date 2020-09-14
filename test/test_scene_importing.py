from gibson2.simulator import Simulator
from gibson2.scenes.stadium_scene import StadiumScene
from gibson2.scenes.indoor_scene import IndoorScene
from gibson2.robots.robot_locomotors import Turtlebot
from gibson2.utils.utils import parse_config
import os
import gibson2

from gibson2.utils.assets_utils import download_assets, download_demo_data

download_assets()
download_demo_data()
config = parse_config(os.path.join(gibson2.root_path, '../test/test.yaml'))

def test_import_building():
    s = Simulator(mode='headless')
    scene = IndoorScene('Rs')
    s.import_scene(scene, texture_scale=0.4)
    for i in range(15):
        s.step()
    assert s.objects == list(range(2))
    s.disconnect()


def test_import_building_big():
    s = Simulator(mode='headless')
    scene = IndoorScene('Rs')
    s.import_scene(scene, texture_scale=1)
    assert s.objects == list(range(2))
    s.disconnect()


def test_import_stadium():
    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    print(s.objects)
    assert s.objects == list(range(4))
    s.disconnect()


def test_import_building_viewing():
    s = Simulator(mode='headless')
    scene = IndoorScene('Rs')
    s.import_scene(scene)
    assert s.objects == list(range(2))

    turtlebot1 = Turtlebot(config)
    turtlebot2 = Turtlebot(config)
    turtlebot3 = Turtlebot(config)

    s.import_robot(turtlebot1)
    s.import_robot(turtlebot2)
    s.import_robot(turtlebot3)

    turtlebot1.set_position([0.5, 0, 0.5])
    turtlebot2.set_position([0, 0, 0.5])
    turtlebot3.set_position([-0.5, 0, 0.5])

    for i in range(10):
        s.step()
        #turtlebot1.apply_action(np.random.randint(4))
        #turtlebot2.apply_action(np.random.randint(4))
        #turtlebot3.apply_action(np.random.randint(4))

    s.disconnect()
