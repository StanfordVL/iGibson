import os

import igibson
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_model_path
from igibson.utils.utils import parse_config


def main():
    config = parse_config(os.path.join(igibson.example_config_path, "turtlebot_demo.yaml"))
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, optimized=True)
    s = Simulator(mode="gui", image_width=512, image_height=512, rendering_settings=settings)

    scene = EmptyScene()
    s.import_scene(scene)

    apple_dir = get_ig_model_path("apple", "00_0")
    apple_urdf = os.path.join(get_ig_model_path("apple", "00_0"), "00_0.urdf")

    apple = URDFObject(apple_urdf, name="apple", category="apple", model_path=apple_dir)

    s.import_object(apple)
    apple.set_position([0, 0, 0.2])
    temp = 0
    for i in range(10000):
        if i % 10 == 0:
            temp += 1
        print(temp)
        apple.states[object_states.Temperature].set_value(temp)
        print(apple.states[object_states.Cooked].get_value())
        with Profiler("Simulator step"):
            s.step()
    s.disconnect()


if __name__ == "__main__":
    main()
