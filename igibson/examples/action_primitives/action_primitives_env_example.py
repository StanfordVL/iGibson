import logging
import os
import platform

import numpy as np

import igibson
from igibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitive
from igibson.envs.action_primitive_env import ActionPrimitivesEnv
from igibson.tasks.dummy_task import DummyTask
from igibson.utils.utils import parse_config


def main(selection="user", headless=False, short_exec=False):
    """
    Creates an action primitive environment and showcases the NAVIGATE_TO primitive from the starter semantic action
    primitives.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)
    config_filename = os.path.join(igibson.configs_path, "behavior_robot_mp_behavior_task.yaml")
    config = parse_config(config_filename)
    config["not_load_object_categories"] = ["ceilings"]
    config["load_object_categories"] = [
        "walls",
        "floors",
        "bottom_cabinet",
        "top_cabinet",
        "door",
        "sink",
        "coffee_table",
        "fridge",
        "countertop",
        "bath_towel",
        "soap",
        "bowl",
        "cup",
    ]
    env = ActionPrimitivesEnv(
        "StarterSemanticActionPrimitives",
        config_file=config,
        mode="gui_interactive" if not headless else "headless",
        use_pb_gui=(not headless and platform.system() != "Darwin"),
    )

    env.env.task = DummyTask(env.env)

    if env.env.simulator.viewer is not None:
        env.env.simulator.viewer.following_viewer = True
        env.env.simulator.viewer.camlocation_in_rf = np.array([1.0, 0.0, 1])  # x is in front of the robot

    for _ in range(10):
        env.env.simulator.step()
        env.env.simulator.sync(force_sync=True)

    # Pick an object to navigate to.
    obj = env.scene.objects_by_category["sink"][1]

    # Convert the human-interpretable (Primitive, Object) pair to the env-compatible (PrimitiveId, ObjectId) pair.
    action = env.action_generator.get_action_from_primitive_and_object(StarterSemanticActionPrimitive.NAVIGATE_TO, obj)

    # Pass the action into the environment. This will start an attempt to plan and execute the primitive action.
    env.step(action)

    # Close the environment when the primitive has executed.
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    logging.getLogger(name="igibson.action_primitives.starter_semantic_action_primitives").setLevel(level=logging.DEBUG)
    logging.getLogger(name="igibson.utils.motion_planning_utils").setLevel(level=logging.DEBUG)
    main()
