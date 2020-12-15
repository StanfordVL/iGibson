from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.utils.utils import parse_config
import os
import gibson2
from gibson2.utils.assets_utils import download_assets, download_demo_data
import time
import matplotlib.pyplot as plt


class DemoInteractive(object):
    def __init__(self):
        download_assets()
        download_demo_data()

    def run_demo(self):
        config = parse_config(os.path.join(gibson2.assets_path, 'example_configs/turtlebot_demo.yaml'))
        s = Simulator(mode='gui', image_width=700, image_height=700)
        scene = StaticIndoorScene('Rs_interactive')
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