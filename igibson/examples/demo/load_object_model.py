""" 
This demo shows how to load a scaled object from the model library
"""
import os

import numpy as np
import pybullet as p
import pybullet_data

import igibson
from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path


def main():
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )
    s = Simulator(mode="gui_interactive", rendering_settings=vr_rendering_settings, image_height=512, image_width=512)

    scene = EmptyScene()
    # scene.load_object_categories(benchmark_names)

    s.import_scene(scene, render_floor_plane=True)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    vr_agent = BehaviorRobot(s)
    s.import_behavior_robot(vr_agent)
    s.register_main_vr_robot(vr_agent)
    vr_agent.activate()

    table_objects_to_load = {
        "table_1": {
            "category": "breakfast_table",
            "model": "1b4e6f9dd22a8c628ef9d976af675b86",
            "pos": (1.000000, -0.200000, 1.01),
            "orn": (0, 0, 90),
        },
        "table_2": {
            "category": "breakfast_table",
            "model": "1b4e6f9dd22a8c628ef9d976af675b86",
            "pos": (-1.500000, -3.000000, 1.01),
            "orn": (0, 0, 90),
        },
    }

    avg_category_spec = get_ig_avg_category_specs()

    scene_objects = {}
    for obj in table_objects_to_load.values():
        category = obj["category"]
        if category in scene_objects:
            scene_objects[category] += 1
        else:
            scene_objects[category] = 1

        category_path = get_ig_category_path(category)
        if "model" in obj:
            model = obj["model"]
        else:
            model = np.random.choice(os.listdir(category_path))
        model_path = get_ig_model_path(category, model)
        filename = os.path.join(model_path, model + ".urdf")
        obj_name = "{}_{}".format(category, scene_objects[category])

        simulator_obj = URDFObject(
            filename,
            name=obj_name,
            category=category,
            model_path=model_path,
            avg_obj_dims=avg_category_spec.get(category),
            fit_avg_dim_volume=True,
            texture_randomization=False,
            overwrite_inertial=True,
            initial_pos=obj["pos"],
            initial_orn=obj["orn"],
        )
        s.import_object(simulator_obj)

    while True:
        s.step()

    s.disconnect()


if __name__ == "__main__":
    main()
