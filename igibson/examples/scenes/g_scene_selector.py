import logging
from sys import platform

import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_available_g_scenes
from igibson.utils.utils import let_user_pick


def main(selection="user", headless=False, short_exec=False):
    """
    Prompts the user to select any available non-interactive scene and loads it.
    Shows how to load directly scenes without the Environment interface
    Shows how to sample points in the scene and how to compute geodesic distance and the shortest path
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    available_g_scenes = get_first_options()
    scene_id = available_g_scenes[let_user_pick(available_g_scenes, selection=selection) - 1]
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    # Reduce texture scale for Mac.
    if platform == "darwin":
        settings.texture_scale = 0.5
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )

    scene = StaticIndoorScene(
        scene_id,
        build_graph=True,
    )
    s.import_scene(scene)

    # Shows how to sample points in the scene
    np.random.seed(0)
    for _ in range(10):
        random_floor = scene.get_random_floor()
        p1 = scene.get_random_point(random_floor)[1]
        p2 = scene.get_random_point(random_floor)[1]
        shortest_path, geodesic_distance = scene.get_shortest_path(random_floor, p1[:2], p2[:2], entire_path=True)
        print("Random point 1: {}".format(p1))
        print("Random point 2: {}".format(p2))
        print("Geodesic distance between p1 and p2: {}".format(geodesic_distance))
        print("Shortest path from p1 to p2: {}".format(shortest_path))

    if not headless:
        input("Press enter")

    max_steps = -1 if not short_exec else 1000
    step = 0
    while step != max_steps:
        with Profiler("Simulator step"):
            s.step()
            step += 1

    s.disconnect()


def get_first_options():
    return get_available_g_scenes()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
