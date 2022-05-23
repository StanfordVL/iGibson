import logging
import os
import platform

import numpy as np

import igibson
from igibson.action_primitives.b1k_discrete_action_primitives import B1KActionPrimitives
from igibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitive
from igibson.envs.action_primitive_env import ActionPrimitivesEnv
from igibson.utils.utils import parse_config


def main(selection="user", headless=False, short_exec=False):
    """
    Creates an action primitive environment and showcases the several primitive from the b1k action primitives set
    primitives.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(igibson.configs_path, "fetch_behavior_aps.yaml")
    config = parse_config(config_filename)
    env = ActionPrimitivesEnv(
        "B1KActionPrimitives",
        config_file=config,
        mode="headless" if headless else "gui_interactive",
        use_pb_gui=(not headless and platform.system() != "Darwin"),
    )

    # Change this if you want to visualize full paths
    full_base_path = False
    full_arm_path = False
    env.action_generator.skip_base_planning = not full_base_path
    env.action_generator.skip_arm_planning = not full_arm_path

    # Make the viewer follow the robot, placing the virtual camera in front of it and watching it
    if env.env.simulator.viewer is not None:
        env.env.simulator.viewer.following_viewer = True
        env.env.simulator.viewer.camlocation_in_rf = np.array([1.0, 0.0, 2.2])  # x is in front of the robot

    env.task.initial_state = env.task.save_scene(env)

    env.reset()
    if env.env.config["task"] in ["putting_away_Halloween_decorations"]:
        env.env.scene.open_all_objs_by_category(category="bottom_cabinet", mode="value", value=0.05)
        print("bottom_cabinet opened!")
        if env.env.robots[0].model_name == "Tiago":
            # Change its initial location because it collides
            robot_position, robot_orn = env.env.robots[0].get_position_orientation()
            env.env.robots[0].set_position_orientation(np.array([0.5, 0, robot_position[2]]), robot_orn)
            print("robot moved!")

    # Pass the action into the environment. This will start an attempt to plan and execute the primitive action.
    for action_idx in [0, 1, 2, 3, 0, 4, 5, 6, 0, 4, 7]:
        o, r, d, i = env.step(action_idx)
        if i["primitive_success"]:
            continue
        else:
            print("Primitive {} failed. Ending".format(action_idx))
            break

    # Close the environment when the primitive has executed.
    env.close()
    print("End")


if __name__ == "__main__":
    # Change the level to logging.DEBUG logging.INFO logging.WARNING ... depending on how much you want to see
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(name="igibson").setLevel(level=logging.INFO)
    logging.getLogger(name="igibson.action_primitives").setLevel(level=logging.INFO)
    logging.getLogger(name="igibson.utils.motion_planning_utils").setLevel(level=logging.INFO)
    logging.getLogger(name="igibson.external.pybullet_tools.utils").setLevel(level=logging.INFO)
    main()
