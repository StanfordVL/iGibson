import logging
import os
import platform

import igibson
from igibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitive
from igibson.envs.action_primitive_env import ActionPrimitivesEnv
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
    env = ActionPrimitivesEnv(
        "StarterSemanticActionPrimitives",
        config_file=config,
        mode="headless" if headless else "gui_non_interactive",
        use_pb_gui=(not headless and platform.system() != "Darwin"),
    )

    # Pick an object to navigate to.
    obj = env.scene.objects_by_category["sink"][0]

    # Convert the human-interpretable (Primitive, Object) pair to the env-compatible (PrimitiveId, ObjectId) pair.
    action = env.action_generator.get_action_from_primitive_and_object(StarterSemanticActionPrimitive.NAVIGATE_TO, obj)

    # Pass the action into the environment. This will start an attempt to plan and execute the primitive action.
    env.step(action)

    # Close the environment when the primitive has executed.
    env.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    main()
