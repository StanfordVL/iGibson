""" 
This demo shows how to load a scaled object from the model library
"""
import os

import numpy as np

from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.robots.behavior_robot import BehaviorRobot
# from igibson.scenes.empty_scene import EmptyScene
from igibson.scenes.igibson_indoor_scene import HomeScene
from igibson.scenes.scene_library import get_scene_path
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path


def main():
    # VR rendering settings
    mesh_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )
    s = Simulator(mode="gui_interactive", rendering_settings=mesh_rendering_settings, image_height=512, image_width=512)
    urdf, scene_dir = get_scene_path("Rs_int", "IG")
    # scene = EmptyScene()
    scene = HomeScene(urdf=urdf, scene_dir=scene_dir, traversal_map_settings={}, scene_settings = {})

    s.import_ig_scene(scene)

    while True:
        s.step()


if __name__ == "__main__":
    main()
