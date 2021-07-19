import argparse
import glob
import json
import os
import time

import igibson
from igibson.envs.behavior_mp_env import ActionPrimitives, BehaviorMPEnv


def get_empty_hand(current_hands):
    if len(current_hands) == 0:
        return "right_hand"
    elif len(current_hands) == 1:
        return "left_hand" if next(current_hands.values()) == "right_hand" else "right_hand"

    raise ValueError("Both hands are full but you are trying to execute a grasp.")


def get_actions_from_segmentation(env, demo_data):
    hand_by_object = {}
    actions = []
    segmentation = demo_data["segmentations"]["flat"]["sub_segments"]
    for segment in segmentation:
        state_records = [sr for sr in segment["state_records"] if all("room_floor" not in obj for obj in sr["objects"])]
        if len(state_records) == 0:
            print("Found segment with no useful state changes: %r" % segment)
            continue
        elif len(state_records) > 1:
            print("Found segment with multiple state changes, using the first: %r" % segment)

        state_change = state_records[0]

        # Get the objects' indices.
        object_idxes = []
        for obj_name in state_change["objects"]:
            # Find the object in the scope.
            obj = env.task.simulator.scene.objects_by_name[obj_name]
            idx = env.task_relevant_objects.index(obj)
            object_idxes.append(idx)

        # Handle the combinations that we support.
        state_name = state_change["name"]
        state_value = state_change["value"]

        if state_name == "Open" and state_value == True:
            primitive = ActionPrimitives.OPEN
            target_object = object_idxes[0]
        elif state_name == "Open" and state_value == False:
            primitive = ActionPrimitives.CLOSE
            target_object = object_idxes[0]
        # We will support automatic navigation to relevant objects.
        elif state_name == "InReachOfRobot" and state_value == True:
            pass
        #     primitive = ActionPrimitives.NAVIGATE_TO
        #     object_in_position = 0
        elif state_name == "InHandOfRobot" and state_value == True:
            target_object = object_idxes[0]
            hand = get_empty_hand(hand_by_object)
            hand_by_object[target_object] = hand
            primitive = ActionPrimitives.LEFT_GRASP if hand == "left_hand" else ActionPrimitives.RIGHT_GRASP
        elif state_name == "Inside" and state_value == True:
            target_object = object_idxes[1]
            assert target_object in hand_by_object, "Placed object not currently grasped."
            hand = hand_by_object[target_object]
            primitive = (
                ActionPrimitives.LEFT_PLACE_INSIDE if hand == "left_hand" else ActionPrimitives.RIGHT_PLACE_INSIDE
            )
        elif state_name == "OnTop" and state_value == True:
            target_object = object_idxes[1]
            assert target_object in hand_by_object, "Placed object not currently grasped."
            hand = hand_by_object[target_object]
            primitive = ActionPrimitives.LEFT_PLACE_ONTOP if hand == "left_hand" else ActionPrimitives.RIGHT_PLACE_ONTOP
        else:
            raise ValueError("Found a state change we can't process: %r" % state_change)

        # Append the action.
        action = int(primitive) * env.num_objects + target_object
        actions.append(action)

    return actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        "-c",
        default=os.path.join(igibson.example_config_path, "behavior/behavior_segmentation_replay.yaml"),
        help="which config file to use [default: use yaml files in examples/configs]",
    )
    parser.add_argument(
        "--mode",
        "-m",
        choices=["headless", "gui", "iggui", "pbgui"],
        default="gui",
        help="which mode for simulation (default: headless)",
    )
    args = parser.parse_args()

    env = BehaviorMPEnv(
        config_file=args.config,
        mode=args.mode,
        action_timestep=1.0 / 300.0,
        physics_timestep=1.0 / 300.0,
        use_motion_planning=False,
    )

    # Load the segmentation of a demo for this task.
    data_path = r"C:\Users\cgokmen\Stanford Drive\BEHAVIOR resources\segmentation_results"
    file_pattern = "{}_{}_{}_*.json".format(env.config["task"], env.config["task_id"], env.config["scene_id"])
    options = sorted(glob.glob(os.path.join(data_path, file_pattern)))

    with open(options[0], "r") as f:
        selected_demo_data = json.load(f)

    # Get the actions from the segmentation
    actions = get_actions_from_segmentation(env, selected_demo_data)

    start = time.time()
    env.reset()

    # env.robots[0].set_position_orientation([0, 0, 0.7], [0, 0, 0, 1])
    done = False
    for action in actions:  # 10 seconds
        state, reward, done, info = env.step(action)
        print(reward, info)
        if done:
            break
    print(
        "Episode finished after {} timesteps, took {} seconds. Done: {}".format(
            env.current_step, time.time() - start, done
        )
    )
    env.close()


if __name__ == "__main__":
    main()
