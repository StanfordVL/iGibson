from igibson.robots.turtlebot_robot import Turtlebot
from igibson.simulator import Simulator
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.utils.utils import parse_config
import os
import igibson
from igibson.utils.assets_utils import download_assets, download_demo_data


class DemoStatic(object):
    def __init__(self):
        download_assets()
        download_demo_data()

    def run_demo(self):
        config = parse_config(os.path.join(igibson.example_config_path,
            'turtlebot_demo.yaml'))

        s = Simulator(mode='gui', image_width=700, image_height=700)
        scene = StaticIndoorScene('Rs', pybullet_load_texture=True)
        s.import_scene(scene)
        turtlebot = Turtlebot(config)
        s.import_robot(turtlebot)

        for i in range(1000):
            turtlebot.apply_action([0.1, 0.5])
            s.step()

        s.disconnect()


if __name__ == "__main__":
    DemoStatic().run_demo()
