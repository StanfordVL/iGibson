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

    scene = EmptyScene()
    s.scene = scene
    scene.objects_by_id = {}

    s.import_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    agent = REGISTERED_ROBOTS[args.robot](action_type="continuous")

    s.import_robot(agent)

    # Make sure robot is a two wheeled robot and using differential drive
    assert isinstance(agent, TwoWheelRobot), "Robot must be a TwoWheelRobot!"
    assert (
        agent.controller_config["base"]["name"] == "DifferentialDriveController"
    ), "Robot must be using differential drive control for its base"

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
