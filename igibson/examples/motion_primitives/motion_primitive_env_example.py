import os

import igibson
from igibson.motion_primitives import BehaviorMotionPrimitiveEnv, MotionPrimitive


def main():
    config_filename = os.path.join(igibson.example_config_path, "behavior_full_observability.yaml")
    env = BehaviorMotionPrimitiveEnv(activity_relevant_objects_only=False, config_file=config_filename, mode="pbgui")

    obj = env.scene.objects_by_category["sink"][0]
    env.human_readable_step(MotionPrimitive.NAVIGATE_TO, obj)
    env.close()


if __name__ == "__main__":
    main()
