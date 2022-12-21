import logging
import os

import numpy as np

import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config
import ipdb
import pybullet as p

def main(selection="user", headless=False, short_exec=False):
    """
    Robot demo
    Loads all robots in an empty scene, generate random actions
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # Create empty scene
    settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        rendering_settings=settings,
        use_pb_gui=True,
    )
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Create one instance of each robot aligned along the y axis
    position = [0, 0, 0]
    robots = {}

    config = parse_config(os.path.join(igibson.configs_path, "robots", "hsr.yaml"))
    robot_config = config["robot"]
    robot_name = robot_config.pop("name")
    robot = REGISTERED_ROBOTS[robot_name](**robot_config)
    s.import_object(robot)
    robot.set_position(position)
    robot.reset()
    robot.keep_still()

    robots[robot_name] = (robot, position[1])
    print("Loaded " + robot_name)

    if not headless:
        # Set viewer in front
        s.viewer.initial_pos = [1.6, 0, 1.3]
        s.viewer.initial_view_direction = [-0.7, 0, -0.7]
        s.viewer.reset_viewer()

    examine_self_collision = False
    if examine_self_collision:
        # Obtain the correct listing of collisions
        link_idx_2_name = {}
        for key in robot.links.keys():
            link_idx_2_name[robot.links[key].link_id] = key

    for i in range(100):  # keep still for 10 seconds
        s.step()
        collision_links = [
            collision for bid in robot.get_body_ids() for collision in p.getContactPoints(bodyA=bid)
        ]

        if examine_self_collision:
            print("printing out collisions")
            for link in collision_links:
                if link[2] == 3:
                    print([link_idx_2_name[link[3]], link_idx_2_name[link[4]]])
            print(f"colliding links: {[(link[3], link[4]) for link in collision_links]}")

    print("Moving " + robot_name)

    for _ in range(300):
        action = np.random.uniform(-1, 1, robot.action_dim)
        robot.apply_action(action)
        for _ in range(10):
            s.step()

    robot.keep_still()
    s.reload()
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
