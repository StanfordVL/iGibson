import os

import IPython

import igibson
from igibson import object_states
from igibson.examples.mp_replay.behavior_motion_primitive_env import BehaviorMotionPrimitiveEnv, MotionPrimitive


def main():
    config_filename = os.path.join(igibson.example_config_path, "behavior_full_observability.yaml")
    env = BehaviorMotionPrimitiveEnv(activity_relevant_objects_only=False, config_file=config_filename, mode="pbgui")

    # TODO: Fix this once we become able to open doors.
    for door in env.scene.objects_by_category["door"]:
        door.states[object_states.Open].set_value(True, fully=True)

    # obj = env.scene.objects_by_category["sink"][0]
    obj = env.scene.objects_by_category["sink"][0]
    env.human_readable_step(MotionPrimitive.NAVIGATE_TO, obj)
    env.close()


if __name__ == "__main__":
    main()
