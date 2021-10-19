"""This file contains utils for BEHAVIOR demo replay checkpoints."""
import json
import os

import pybullet as p


def save_task_relevant_state(env, root_directory, filename="behavior_dump"):
    json_path = os.path.join(root_directory, f"{filename}.json")

    # Save the dump in a file.
    with open(json_path, "w") as f:
        json.dump(save_task_relevant_object_and_robot_states(env), f)

def save_sim_urdf_object_state(sim, root_directory, filename="behavior_dump"):
    json_path = os.path.join(root_directory, f"{filename}.json")

    # Save the dump in a file.
    with open(json_path, "w") as f:
        json.dump(save_all_scene_object_and_robot_states(sim), f)

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


def save_task_relevant_object_and_robot_states(env):
    # Dump the object state.
    object_dump = {}
    for obj in env.task_relevant_objects:
        object_dump[obj.bddl_object_scope] = {'metadata': obj.metadata, 'asset_path': obj.model_path, 'pose': tuple(obj.get_position_orientation()), 'scale': tuple(obj.scale)}

    # Dump the robot state.
    robot_dump = []
    for robot in env.simulator.robots:
        robot_dump.append(robot.dump_state())

    return {"objects": object_dump, "robots": robot_dump}

def save_all_scene_object_and_robot_states(sim):
    # Dump the object state, but only for objects of type URDFObject 
    # that are in the sim.
    object_dump = {}
    for obj in sim.scene.get_objects():
        if 'URDFObject' in str(type(obj)):
            object_dump[obj.bddl_object_scope] = {'metadata': obj.metadata, 'asset_path': obj.model_path, 'pose': tuple(obj.get_position_orientation()), 'scale': tuple(obj.scale)}

    # Dump the robot state.
    robot_dump = []
    for robot in sim.robots:
        robot_dump.append(robot.dump_state())

    return {"objects": object_dump, "robots": robot_dump}