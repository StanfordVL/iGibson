"""This file contains utils for BEHAVIOR demo replay checkpoints."""
import json
import os

import pybullet as p


def save_checkpoint(simulator, root_directory):
    bullet_path = os.path.join(root_directory, "%d.bullet" % simulator.frame_count)
    json_path = os.path.join(root_directory, "%d.json" % simulator.frame_count)

    # Save the simulation state.
    p.saveBullet(bullet_path)

    # Save the dump in a file.
    with open(json_path, "w") as f:
        json.dump(save_internal_states(simulator), f)


def load_checkpoint(simulator, root_directory, frame):
    bullet_path = os.path.join(root_directory, "%d.bullet" % frame)
    json_path = os.path.join(root_directory, "%d.json" % frame)

    # Restore the simulation state.
    p.restoreState(fileName=bullet_path)

    with open(json_path, "r") as f:
        dump = json.load(f)

    load_internal_states(simulator, dump)


def save_internal_states(simulator):
    # Dump the object state.
    object_dump = {}
    for name, obj in simulator.scene.objects_by_name.items():
        object_dump[name] = obj.dump_state()

    # Dump the robot state.
    robot_dump = []
    for robot in simulator.robots:
        robot_dump.append(robot.dump_state())

    return {"objects": object_dump, "robots": robot_dump}


def load_internal_states(simulator, dump):
    # Restore the object state.
    object_dump = dump["objects"]
    for name, obj in simulator.scene.objects_by_name.items():
        obj.load_state(object_dump[name])

    # Restore the robot state.
    robot_dumps = dump["robots"]
    for robot, robot_dump in zip(simulator.robots, robot_dumps):
        robot.load_state(robot_dump)
