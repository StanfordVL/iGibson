import os

import igibson
from igibson.action_generators.motion_primitive_generator import MotionPrimitive, MotionPrimitiveActionGenerator
from igibson.envs.action_generator_env import ActionGeneratorEnv


def main():
    config_filename = os.path.join(igibson.example_config_path, "behavior_full_observability.yaml")
    env = ActionGeneratorEnv(
        MotionPrimitiveActionGenerator, config_file=config_filename, mode="gui_non_interactive", use_pb_gui=True
    )

    obj = env.scene.objects_by_category["sink"][0]
    action = env.action_generator.get_action_from_primitive_and_object(MotionPrimitive.NAVIGATE_TO, obj)
    env.step(action)
    env.close()


if __name__ == "__main__":
    main()
