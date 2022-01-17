import logging
from sys import platform

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main():
    """
    Example of partial loading of a scene
    Loads only some objects (by category) and in some room types
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    if platform == "darwin":
        settings.texture_scale = 0.5
    s = Simulator(mode="gui_interactive", image_width=512, image_height=512, rendering_settings=settings)
    scene = InteractiveIndoorScene(
        "Rs_int",
        texture_randomization=False,
        object_randomization=False,
        load_object_categories=["swivel_chair"],
        load_room_types=["living_room"],
    )
    s.import_scene(scene)

    while True:
        s.step()


if __name__ == "__main__":
    main()
