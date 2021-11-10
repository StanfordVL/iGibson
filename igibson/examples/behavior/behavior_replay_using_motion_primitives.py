import argparse
import json
import os
import time

import yaml

import igibson
from igibson.envs.behavior_mp_env import ActionPrimitives, BehaviorMPEnv
from igibson.utils.ig_logging import IGLogReader


def get_empty_hand(current_hands):
    if len(current_hands) == 0:
        return "right_hand"
    elif len(current_hands) == 1:
        return "left_hand" if list(current_hands.values())[0] == "right_hand" else "right_hand"

    raise ValueError("Both hands are full but you are trying to execute a grasp.")


def get_actions_from_segmentation(demo_data):
    print("Conversion of demo segmentation to motion primitives:")

    hand_by_object = {}
    actions = []
    segmentation = demo_data["segmentations"]["flat"]["sub_segments"]

    # Convert the segmentation to a sequence of state changes.
    state_changes = []
    for segment in segmentation:
        state_records = segment["state_records"]
        if len(state_records) == 0:
            print("Found segment with no useful state changes: %r" % segment)
            continue
        elif len(state_records) > 1:
            print("Found segment with multiple state changes, using the first: %r" % segment)

        state_change = state_records[0]
        state_changes.append(state_change)

    # Now go through the state changes and convert them to actions
    for i, state_change in enumerate(state_changes):
        # Handle the combinations that we support.
        state_name = state_change["name"]
        state_value = state_change["value"]

        # TODO(replayMP): Here we compute grasps based on the InHand state. Ditch this and simply do a single-hand
        # grasp on the object we will manipulate next. That way it will be fetch-compatible.
        if state_name == "Open" and state_value is True:
            primitive = ActionPrimitives.OPEN
            target_object = state_change["objects"][0]
        elif state_name == "Open" and state_value is False:
            primitive = ActionPrimitives.CLOSE
            target_object = state_change["objects"][0]
        elif state_name == "InReachOfRobot" and state_value is True:
            # The primitives support automatic navigation to relevant objects.
            continue
        elif state_name == "InHandOfRobot" and state_value is True:
            target_object = state_change["objects"][0]

            # Check that we do something with this object later on, otherwise don't grasp it.
            is_used = False
            for future_state_change in state_changes[i + 1 :]:
                # We should only grasp the moved object in these cases.
                if future_state_change["objects"][0] != target_object:
                    continue

                if future_state_change["name"] == "InHandOfRobot" and future_state_change["value"] is True:
                    # This object is re-grasped later. No need to look any further than that.
                    break

                # We only care about Inside and OnTop use cases later.
                if future_state_change["name"] not in ("Inside", "OnTop") or future_state_change["value"] is False:
                    continue

                # This is a supported use case so we approve the grasp.
                is_used = True
                break

            # If the object is not used in the future, don't grasp it.
            if not is_used:
                continue

            hand = get_empty_hand(hand_by_object)
            hand_by_object[target_object] = hand
            primitive = ActionPrimitives.LEFT_GRASP if hand == "left_hand" else ActionPrimitives.RIGHT_GRASP
        elif state_name == "Inside" and state_value is True:
            placed_object = state_change["objects"][0]
            target_object = state_change["objects"][1]
            if placed_object not in hand_by_object:
                print(
                    "Placed object %s in segment %d not currently grasped. Maybe some sort of segmentation error?"
                    % (placed_object, i)
                )
                continue
            hand = hand_by_object[placed_object]
            del hand_by_object[placed_object]
            primitive = (
                ActionPrimitives.LEFT_PLACE_INSIDE if hand == "left_hand" else ActionPrimitives.RIGHT_PLACE_INSIDE
            )
        elif state_name == "OnTop" and state_value is True:
            placed_object = state_change["objects"][0]
            target_object = state_change["objects"][1]
            if placed_object not in hand_by_object:
                print(
                    "Placed object %s in segment %d not currently grasped. Maybe some sort of segmentation error?"
                    % (placed_object, i)
                )
                continue
            hand = hand_by_object[placed_object]
            del hand_by_object[placed_object]
            primitive = ActionPrimitives.LEFT_PLACE_ONTOP if hand == "left_hand" else ActionPrimitives.RIGHT_PLACE_ONTOP
        else:
            raise ValueError("Found a state change we can't process: %r" % state_change)

        # Append the action.
        action = (primitive, target_object)
        actions.append(action)
        print("%s(%s)" % action)

    print("Conversion complete.\n")
    return actions


def run_demonstration(demo_path, segmentation_path, output_path):
    task = IGLogReader.read_metadata_attr(demo_path, "/metadata/atus_activity")
    task_id = IGLogReader.read_metadata_attr(demo_path, "/metadata/activity_definition")
    scene_id = IGLogReader.read_metadata_attr(demo_path, "/metadata/scene_id")

    # Load the segmentation of a demo for this task.
    with open(segmentation_path, "r") as f:
        selected_demo_data = json.load(f)

    # Get the actions from the segmentation
    actions = get_actions_from_segmentation(selected_demo_data)

    # Prepare the environment
    config_file = os.path.join(igibson.example_config_path, "behavior_segmentation_replay.yaml")
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    config["task"] = task
    config["task_id"] = task_id
    config["scene_id"] = scene_id

    env = BehaviorMPEnv(
        config_file=config,
        mode="headless",
        action_timestep=1.0 / 300.0,
        physics_timestep=1.0 / 300.0,
        use_motion_planning=False,
        activity_relevant_objects_only=False,
    )

    start = time.time()
    env.reset()

    # env.robots[0].set_position_orientation([0, 0, 0.7], [0, 0, 0, 1])
    done = False
    infos = []
    action_successes = []
    import pybullet as p

    p.configureDebugVisualizer(p.COV_ENABLE_GUI, False)
    p.resetDebugVisualizerCamera(cameraTargetPosition=[1, -1, 0], cameraDistance=4, cameraYaw=240, cameraPitch=-45)
    for action_pair in actions:
        # try:
        print("Executing %s(%s)" % action_pair)
        primitive, obj_name = action_pair

        # Convert the action
        obj_id = next(i for i, obj in enumerate(env.addressable_objects) if obj.name == obj_name)
        action = int(primitive) * env.num_objects + obj_id

        # Execute.
        state, reward, done, info = env.step(action)
        print(reward, info)
        infos.append(info)
        action_successes.append(True)
        if done:
            break
        # except:
        #     action_successes.append(False)

    # Dump the results
    data = {"actions": actions, "infos": infos, "action_successes": action_successes}
    with open(output_path, "w") as f:
        json.dump(data, f)

    print(
        "Episode finished after {} timesteps, took {} seconds. Done: {}".format(
            env.current_step, time.time() - start, done
        )
    )
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("demo_path", type=str, help="Path of the demo hdf5 to replay.")
    parser.add_argument("segmentation_path", type=str, help="Path of the segmentation of the demo.")
    parser.add_argument("output_path", type=str, help="Path to output result JSON file to.")
    args = parser.parse_args()

    run_demonstration(args.demo_path, args.segmentation_path, args.output_path)


if __name__ == "__main__":
    main()
