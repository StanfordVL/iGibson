import logging
import os
import platform

import igibson
from igibson.action_primitives.b1k_discrete_action_primitives import B1KActionPrimitives
from igibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitive
from igibson.envs.action_primitive_env import ActionPrimitivesEnv
from igibson.utils.utils import parse_config


def main(selection="user", headless=False, short_exec=False):
    """
    Creates an action primitive environment and showcases the NAVIGATE_TO primitive from the starter semantic action
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

    env.task.initial_state = env.task.save_scene(env)

    env.reset()
    if env.env.config["task"] in ["putting_away_Halloween_decorations"]:
        env.env.scene.open_all_objs_by_category(category="bottom_cabinet", mode="value", value=0.05)
        print("bottom_cabinet opened!")

    # env.env.simulator.viewer.initial_pos = [1.5, -2.0, 2.3]
    # env.env.simulator.viewer.initial_view_direction = [-0.7, 0.0, -0.6]
    env.env.simulator.viewer.initial_pos = [1.0, -0.3, 1.9]
    env.env.simulator.viewer.initial_view_direction = [-0.1, -0.8, -0.5]
    env.env.simulator.viewer.reset_viewer()

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
    logging.basicConfig(level=logging.DEBUG)
    main()
