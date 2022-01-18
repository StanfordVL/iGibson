import logging

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
from igibson.scenes.stadium_scene import StadiumScene
from igibson.simulator import Simulator


def main():
    """
    Loads the Stadium scene
    This scene is default in pybullet but is not really useful in iGibson
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(mode="gui_interactive", image_width=512, image_height=512, rendering_settings=settings)

    scene = StadiumScene()
    s.import_scene(scene)

    while True:
        with Profiler("Simulator step"):
            s.step()
    s.disconnect()


if __name__ == "__main__":
    main()
