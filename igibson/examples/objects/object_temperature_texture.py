import logging
import os

import igibson
from igibson import object_states
from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_model_path
from igibson.utils.utils import parse_config


def main(selection="user", headless=False, short_exec=False):
    """
    Demo of the texture change for an object
    Loads an apple and increases manually its temperature to observe the texture change from frozen, to cooked, to burnt
    Also shows how to change object-specific parameters such as the burning temperature
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config = parse_config(os.path.join(igibson.configs_path, "turtlebot_static_nav.yaml"))
    settings = MeshRendererSettings(enable_shadow=True, msaa=False, optimized=True)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )

    if not headless:
        # Set a better viewing direction
        s.viewer.initial_pos = [0, 0.4, 0.2]
        s.viewer.initial_view_direction = [-0.4, -0.9, 0]
        s.viewer.reset_viewer()

    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Load and place an apple
    apple_dir = get_ig_model_path("apple", "00_0")
    apple_urdf = os.path.join(get_ig_model_path("apple", "00_0"), "00_0.urdf")
    apple = URDFObject(apple_urdf, name="apple", category="apple", model_path=apple_dir)
    s.import_object(apple)
    apple.set_position([0, 0, 0.2])
    apple.states[object_states.Burnt].burn_temperature = 200

    # Manually increase the temperature of the apple
    for i in range(-10, 100):
        temp = i * 5
        print("Apple temperature: {} degrees Celsius".format(temp))
        apple.states[object_states.Temperature].set_value(temp)
        print("Frozen(Apple)? {}".format(apple.states[object_states.Frozen].get_value()))
        print("Cooked(Apple)? {}".format(apple.states[object_states.Cooked].get_value()))
        print("Burnt(Apple)? {}".format(apple.states[object_states.Burnt].get_value()))
        for j in range(10):
            # with Profiler("Simulator step"):
            s.step()
    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
