from gibson2.robots.robot_locomotors import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.indoor_scene import IndoorScene
from gibson2.utils.utils import parse_config
import os
import gibson2
from gibson2.utils.assets_utils import download_assets, download_demo_data

class DemoInteractive(object):
    def __init__(self):
        download_assets()
        download_demo_data()

    def run_demo(self):
        config = parse_config(os.path.join(gibson2.assets_path, 'example_configs/turtlebot_demo.yaml'))
        s = Simulator(mode='gui', image_width=700, image_height=700)
        scene = IndoorScene('Rs_interactive', is_interactive=True)
        s.import_scene(scene)
        turtlebot = Turtlebot(config)
        s.import_robot(turtlebot)

        for i in range(10000):
            turtlebot.apply_action([0.1,0.5])
            s.step()

        s.disconnect()


if __name__ == "__main__":
    DemoInteractive().run_demo()
