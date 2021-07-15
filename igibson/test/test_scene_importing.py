from igibson.simulator import Simulator
from igibson.scenes.stadium_scene import StadiumScene
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.robots.turtlebot_robot import Turtlebot
from igibson.utils.utils import parse_config
import os
import igibson

from igibson.utils.assets_utils import download_assets, download_demo_data


def test_import_building():
    download_assets()
    download_demo_data()

    s = Simulator(mode='headless')
    scene = StaticIndoorScene('Rs')
    s.import_scene(scene, texture_scale=0.4)
    for i in range(15):
        s.step()
    assert s.objects == list(range(2))
    s.disconnect()


def test_import_building_big():
    download_assets()
    download_demo_data()

    s = Simulator(mode='headless')
    scene = StaticIndoorScene('Rs')
    s.import_scene(scene, texture_scale=1)
    assert s.objects == list(range(2))
    s.disconnect()


def test_import_stadium():
    download_assets()
    download_demo_data()

    s = Simulator(mode='headless')
    scene = StadiumScene()
    s.import_scene(scene)
    print(s.objects)
    assert s.objects == list(range(4))
    s.disconnect()


def test_import_building_viewing():
    download_assets()
    download_demo_data()
    config = parse_config(os.path.join(igibson.root_path, 'test', 'test.yaml'))


    s = Simulator(mode='headless')
    scene = StaticIndoorScene('Rs')
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
