import logging

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main():
    """
    Example of randomization of the texture in a scene
    Loads Rs_int (interactive) and randomizes the texture of the objects
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    settings = MeshRendererSettings(enable_shadow=False, msaa=False)
    s = Simulator(mode="gui_interactive", image_width=512, image_height=512, rendering_settings=settings)
    scene = InteractiveIndoorScene(
        "Rs_int",
        load_object_categories=[],  # To load only the building. Fast
        texture_randomization=True,
        object_randomization=False,
    )
    s.import_scene(scene)

    for i in range(10000):
        if i % 1000 == 0:
            logging.info("Randomize texture")
            scene.randomize_texture()
        s.step()
    s.disconnect()


if __name__ == "__main__":
    main()
