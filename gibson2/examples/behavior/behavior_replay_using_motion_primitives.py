import argparse
import glob
import json
import os
import time

import gibson2
from gibson2.envs.behavior_mp_env import BehaviorMPEnv, ActionPrimitives


def get_actions_from_segmentation(env, demo_data):
    actions = []
    segmentation = demo_data["segmentations"]["flat"]["sub_segments"]
    for segment in segmentation:
        assert len(segment["state_records"]) == 1, "Don't know what to do with multi-change segments."
        state_change = segment['state_records'][0]

        # Get the objects' indices.
        object_idxes = []
        for obj_name in state_change["objects"]:
            # Find the object in the scope.
            obj = env.task.object_scope[obj_name]
            idx = env.task_relevant_objects.index(obj)
            object_idxes.append(idx)

        # Handle the combinations that we support.
        state_name = state_change["name"]
        state_value = state_change["value"]

        if state_name == "Open" and state_value == True:
            primitive = ActionPrimitives.OPEN
            object_in_position = 0
        elif state_name == "Open" and state_value == False:
            primitive = ActionPrimitives.CLOSE
            object_in_position = 0
        elif state_name == "InReachOfRobot" and state_value == True:
            primitive = ActionPrimitives.NAVIGATE_TO
            object_in_position = 0
        elif state_name == "InHandOfRobot" and state_value == True:
            primitive = ActionPrimitives.GRASP
            object_in_position = 0
        elif state_name == "Inside" and state_value == True:
            primitive = ActionPrimitives.PLACE_INSIDE
            object_in_position = 1
        elif state_name == "OnTop" and state_value == True:
            primitive = ActionPrimitives.PLACE_ONTOP
            object_in_position = 1
        else:
            raise ValueError("Found a state change we can't process: %r" % state_change)

        # Append the action.
        object_idx = object_idxes[object_in_position]
        action = int(primitive) * env.num_objects + object_idx
        actions.append(action)

    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        '-c',
        default=os.path.join(gibson2.example_config_path, 'behavior/behavior_collect_misplaced_items.yaml'),
        help='which config file to use [default: use yaml files in examples/configs]')
    parser.add_argument('--mode',
                        '-m',
                        choices=['headless', 'gui', 'iggui', 'pbgui'],
                        default='gui',
                        help='which mode for simulation (default: headless)')
    args = parser.parse_args()

    env = BehaviorMPEnv(config_file=args.config,
                        mode=args.mode,
                        action_timestep=1.0 / 300.0,
                        physics_timestep=1.0 / 300.0,
                        use_motion_planning=True)

    # Load the segmentation of a demo for this task.
    data_path = r"C:\Users\cgokmen\Stanford Drive\BEHAVIOR resources\segmentation_results"
    file_pattern = "{}_{}_{}_*.json".format(env.config["task"], env.config["task_id"], env.config["scene_id"])
    options = sorted(glob.glob(os.path.join(data_path, file_pattern)))

    with open(options[0], "r") as f:
        selected_demo_data = json.load(f)

    # Get the actions from the segmentation
    actions = get_actions_from_segmentation(env, selected_demo_data)

    step_time_list = []
    start = time.time()
    env.reset()

    env.robots[0].set_position_orientation([0, 0, 0.7], [0, 0, 0, 1])
    done = False
    for action in actions:  # 10 seconds
        state, reward, done, info = env.step(action)
        print(reward, info)
        if done:
            break
    print('Episode finished after {} timesteps, took {} seconds. Done: {}'.format(
        env.current_step, time.time() - start, done))
    env.close()


if __name__ == "__main__":
    main()
