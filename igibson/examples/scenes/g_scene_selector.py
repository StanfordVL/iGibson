import logging
from sys import platform

import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_available_g_scenes
from igibson.utils.utils import let_user_pick


def main():
    """
    Prompts the user to select any available non-interactive scene and loads it.
    Shows how to load directly scenes without the Environment interface
    Shows how to sample points in the scene and how to compute geodesic distance and the shortest path
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    available_g_scenes = get_available_g_scenes()
    scene_id = available_g_scenes[let_user_pick(available_g_scenes) - 1]
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    # Reduce texture scale for Mac.
    if platform == "darwin":
        settings.texture_scale = 0.5
    s = Simulator(mode="gui_interactive", image_width=512, image_height=512, rendering_settings=settings)

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
        logging.info("Random point 1: {}".format(p1))
        logging.info("Random point 2: {}".format(p2))
        logging.info("Geodesic distance between p1 and p2: {}".format(geodesic_distance))
        logging.info("Shortest path from p1 to p2: {}".format(shortest_path))

    input("Press enter")

    while True:
        with Profiler("Simulator step"):
            s.step()
    s.disconnect()


if __name__ == "__main__":
    main()
