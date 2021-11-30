"""
This demo shows how to use keyboard to control a Fetch robot via IK
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
from igibson.robots.fetch import Fetch
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_category_path, get_ig_model_path
from igibson.utils.utils import parse_config


def main():
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

    config = parse_config(os.path.join(igibson.root_path, "examples", "configs", "behavior_onboard_sensing_fetch.yaml"))
    vr_agent = Fetch(config)
    s.import_robot(vr_agent)

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

    vr_agent.reset()
    i = 0
    actions = []
    print("Press ctrl-q to quit")
    while True:
        i += 1
        action = np.zeros(11)
        action[-1] = -1.0

        # 0  - forward/backwards (good) (up/down)
        # 1  - rotate robot base (good) (left/right)
        # 2  - turn head left/right (bad) (a/d)
        #     * after reaching end of rotation, causes:
        #     * arm to move, and then body to decrease height -0.1/0.1)
        # 3  - turn head up/down (good) (r/f)
        # 4  - move gripper towards/away from robot (good) (y/h)
        # 5  - move gripper left/right (good) (j/l)
        # 6  - move gripper down/up (bad) (i/k)
        #     * also moves body
        # 7  - rotate gripper clockwise/counter clockwise in Z plane (u/o)
        #     * also moves body
        # 8  - rotate gripper towards/out from body (x plane) ([/')
        #     * also moves body
        # 9  - rotate gripper along plane of floor (y plane)
        #     * also moves body
        # 10 - close/open gripper (good) (z/x)

        events = p.getKeyboardEvents()
        if 65295 in events:
            action[1] += -0.1
        if 65296 in events:
            action[1] += 0.1
        if 65297 in events:
            action[0] += 0.1
        if 65298 in events:
            action[0] += -0.1
        if 97 in events:
            action[2] += 0.1
        if 100 in events:
            action[2] += -0.1
        if 114 in events:
            action[3] += 0.1
        if 102 in events:
            action[3] += -0.1
        if 122 in events:
            action[10] += 0.1
        if 120 in events:
            action[10] += -0.1
        if 105 in events:
            action[6] += 0.1
        if 107 in events:
            action[6] += -0.1
        if 106 in events:
            action[5] += 0.1
        if 108 in events:
            action[5] += -0.1
        if 117 in events:
            action[7] += 0.1
        if 111 in events:
            action[7] += -0.1
        if 121 in events:
            action[4] += 0.1
        if 104 in events:
            action[4] += -0.1
        if 91 in events:
            action[8] += 0.1
        if 39 in events:
            action[8] += -0.1
        if 65307 in events and 113 in events:
            break

        action *= 5
        vr_agent.apply_action(action)
        actions.append(action)
        p.stepSimulation()

    s.disconnect()


if __name__ == "__main__":
    main()
