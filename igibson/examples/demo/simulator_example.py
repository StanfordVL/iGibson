from igibson.robots.turtlebot_robot import Turtlebot
from igibson.simulator import Simulator
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.objects.ycb_object import YCBObject
from igibson.utils.utils import parse_config
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from igibson.render.profiler import Profiler
import igibson
import os

def main():
    config = parse_config(os.path.join(igibson.example_config_path, 'turtlebot_demo.yaml'))
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(mode='gui', image_width=256,
                  image_height=256, rendering_settings=settings)

    scene = StaticIndoorScene('Rs',
                              build_graph=True,
                              pybullet_load_texture=True)
    s.import_scene(scene)
    turtlebot = Turtlebot(config)
    s.import_robot(turtlebot)

    for _ in range(10):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)
        obj.set_position_orientation(np.random.uniform(
            low=0, high=2, size=3), [0, 0, 0, 1])

    print(s.renderer.instances)

    for i in range(10000):
        with Profiler('Simulator step'):
            turtlebot.apply_action([0.1, 0.1])
            s.step()
            rgb = s.renderer.render_robot_cameras(modes=('rgb'))
    s.disconnect()


if __name__ == '__main__':
    main()
