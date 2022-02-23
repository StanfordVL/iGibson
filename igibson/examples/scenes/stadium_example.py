import logging

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.stadium_scene import StadiumScene
from igibson.simulator import Simulator


def main(selection="user", headless=False, short_exec=False):
    """
    Loads the Stadium scene
    This scene is default in pybullet but is not really useful in iGibson
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )

    scene = StadiumScene()
    s.import_scene(scene)

    max_steps = -1 if not short_exec else 1000
    step = 0
    while step != max_steps:
        with Profiler("Simulator step"):
            s.step()
            step += 1
    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
