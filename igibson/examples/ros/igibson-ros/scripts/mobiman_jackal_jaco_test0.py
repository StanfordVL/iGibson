import logging
import os

import numpy as np

import igibson
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.robots import REGISTERED_ROBOTS
from igibson.scenes.empty_scene import EmptyScene
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config


def main(selection="user", headless=False, short_exec=False):
    """
    Robot demo
    Loads all robots in an empty scene, generate random actions
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    # Create empty scene
    # settings = MeshRendererSettings(enable_shadow=False, msaa=False, texture_scale=0.5)
    s = Simulator(
        mode="gui_interactive" if not headless else "headless",
        image_width=512,
        image_height=512,
        # rendering_settings=settings,
        use_pb_gui=True,
        physics_timestep=1/240,
        render_timestep=1/40,
    )
    scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
    s.import_scene(scene)

    # Create one instance of each robot aligned along the y axis
    position = [0, 0, 0]
    robots = {}
    for robot_config_file in os.listdir(os.path.join(igibson.configs_path, "robots")):
        config = parse_config(os.path.join(igibson.configs_path, "robots", robot_config_file))
        robot_config = config["robot"]
        print("[mobiman_jackal_jaco_test0::main] robot_config: " + str(robot_config))
        robot_name = robot_config.pop("name")
        if robot_name != "JackalJaco":
            continue
        robot = REGISTERED_ROBOTS[robot_name](**robot_config)
        s.import_object(robot)
        robot.set_position(position)
        robot.reset()
        robot.keep_still()
        robots[robot_name] = (robot, position[1])
        print("Loaded " + robot_name)
        
        print("Moving " + robot_name)

        if not headless:
            # Set viewer in front
            s.viewer.initial_pos = [1.6, 0, 1.3]
            s.viewer.initial_view_direction = [-0.7, 0, -0.7]
            s.viewer.reset_viewer()
        
        for _ in range(100):  # keep still for 10 seconds
            s.step()
            pass

        for _ in range(30):
            action = np.random.uniform(-1, 1, robot.action_dim)

            action[0] = 0.1
            action[1] = 0.3
            action[2] = 0.1
            action[3] = 0.3
            robot.apply_action(action)
            
            print("[mobiman_jackal_jaco_test0::main] action_dim: " + str(robot.action_dim))
            print("[mobiman_jackal_jaco_test0::main] action: " + str(action))

            for _ in range(100):
                s.step()
                pass

        robot.keep_still()
        s.reload()
        scene = EmptyScene(floor_plane_rgba=[0.6, 0.6, 0.6, 1])
        s.import_scene(scene)

    s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
