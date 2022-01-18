import logging
from sys import platform

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main(random_selection=False, headless=False, short_exec=False):
    """
    Example of randomization of the texture in a scene
    Loads Rs_int (interactive) and randomizes the texture of the objects
    """
    logging.info("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    if platform == "darwin":
        settings.texture_scale = 0.5
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
    )
    load_object_categories = None  # Use the list to load a partial house
    # load_object_categories = [
    #     "bottom_cabinet",
    #     "sofa",
    #     "rocking_chair",
    #     "swivel_chair",
    #     "folding_chair",
    #     "desk",
    #     "dining_table",
    #     "coffee_table",
    #     "breakfast_table",
    # ]  # Use None to load the full house
    scene = InteractiveIndoorScene(
        "Rs_int",
        texture_randomization=False,
        load_object_categories=load_object_categories,
        object_randomization=True,
    )
    s.import_scene(scene)
    loaded = True

    num_resets = 10 if not short_exec else 2
    num_steps_per_reset = 1000 if not short_exec else 10
    total_steps = num_resets * num_steps_per_reset
    for i in range(1, total_steps):
        if i % num_steps_per_reset == 0:
            logging.info("Reloading scene. New objects")
            s.reload()
            loaded = False

        if not loaded:
            logging.info("Randomize object models")

            scene = InteractiveIndoorScene(
                "Rs_int",
                texture_randomization=False,
                load_object_categories=load_object_categories,
                object_randomization=True,
            )
            s.import_scene(scene)
            loaded = True

        s.step()
    s.disconnect()


if __name__ == "__main__":
    main()
