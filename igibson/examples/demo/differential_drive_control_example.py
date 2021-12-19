"""
This demo shows how to use keyboard to control a two-wheeled robot
"""
import argparse
import os
from types import SimpleNamespace

import numpy as np
import pybullet as p
import pybullet_data

import igibson
from igibson.objects.articulated_object import URDFObject
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from igibson.robots import REGISTERED_ROBOTS, TwoWheelRobot
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.utils import parse_config

parser = argparse.ArgumentParser()

parser.add_argument(
    "--robot",
    type=str,
    default="Turtlebot",
    choices=list(REGISTERED_ROBOTS.keys()),
    help="Robot to use. Note that the selected robot must support differential drive!",
)

args = parser.parse_args()


def main(args):
    rendering_settings = MeshRendererSettings(
        optimized=True,
        fullscreen=False,
        enable_shadow=True,
        enable_pbr=True,
        msaa=True,
        light_dimming_factor=1.0,
    )
    s = Simulator(
        mode="headless", use_pb_gui=True, rendering_settings=rendering_settings, image_height=512, image_width=512
    )
    p.configureDebugVisualizer(p.COV_ENABLE_GUI, enable=0)

    scene = EmptyScene()
    s.scene = scene
    scene.objects_by_id = {}

    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    agent = REGISTERED_ROBOTS[args.robot](action_config={"type": "continuous"})

    s.import_robot(agent)

    # Make sure robot is a two wheeled robot and using differential drive
    assert isinstance(agent, TwoWheelRobot), "Robot must be a TwoWheelRobot!"
    assert (
        agent.controller_config["base"]["name"] == "DifferentialDriveController"
    ), "Robot must be using differential drive control for its base"

    table_objects_to_load = {
        "table_1": {
            "category": "breakfast_table",
            "model": "1b4e6f9dd22a8c628ef9d976af675b86",
            "pos": (1.500000, 0.000000, 0.35),
            "orn": (0, 0, 1, 1),
        },
        "coffee_cup_1": {
            "category": "coffee_cup",
            "model": "coffee_cup_000",
            "pos": (1.5, 0.20, 0.8),
            "orn": (0, 0, 0, 1),
        },
        "plate_1": {
            "category": "plate",
            "model": "plate_000",
            "pos": (1.5, -0.20, 0.8),
            "orn": (0, 0, 0, 1),
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
            initial_orn=[0, 0, 90],
        )
        s.import_object(simulator_obj)
        simulator_obj.set_orientation(obj["orn"])

    agent.reset()
    i = 0
    actions = []
    print("Press ctrl-q to quit")
    while True:
        i += 1
        action = np.zeros(agent.action_dim)

        # 0  - forward/backwards (good) (up/down)
        # 1  - rotate robot base (good) (left/right)

        events = p.getKeyboardEvents()
        if 65295 in events:
            action[1] += -0.10
        if 65296 in events:
            action[1] += 0.10
        if 65297 in events:
            action[0] += 0.2
        if 65298 in events:
            action[0] += -0.2
        if 65307 in events and 113 in events:
            break

        agent.apply_action(action)
        actions.append(action)
        p.stepSimulation()

    s.disconnect()


if __name__ == "__main__":
    main(args)
