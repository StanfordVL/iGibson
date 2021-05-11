from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2 import object_states
import numpy as np
from gibson2.render.profiler import Profiler
from gibson2.objects.articulated_object import URDFObject
import gibson2
import os
from gibson2.utils.assets_utils import get_ig_model_path
import pybullet as p


def main():
    config = parse_config(os.path.join(
        gibson2.example_config_path, 'turtlebot_demo.yaml'))
    settings = MeshRendererSettings(
        enable_shadow=False, msaa=False, optimized=True)
    s = Simulator(mode='gui', image_width=512,
                  image_height=512, rendering_settings=settings)

    scene = EmptyScene()
    s.import_scene(scene)

    apple_dir = get_ig_model_path('apple', '00_0')
    apple_urdf = os.path.join(get_ig_model_path('apple', '00_0'), '00_0.urdf')

    apple = URDFObject(apple_urdf, name="apple",
                       category="apple", model_path=apple_dir)

    s.import_object(apple)
    apple.set_position([0, 0, 0.2])
    temp = 0
    for i in range(10000):
        if i % 10 == 0:
            temp += 1
        print(temp)
        apple.states[object_states.Temperature].set_value(temp)
        print(apple.states[object_states.Cooked].get_value())
        with Profiler('Simulator step'):
            s.step()
    s.disconnect()


if __name__ == '__main__':
    main()
