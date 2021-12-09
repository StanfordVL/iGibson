import glob
import json
import logging
import os

import h5py
import networkx as nx
import numpy as np
import tqdm
import yaml

from igibson.envs.igibson_env import iGibsonEnv


def main():
    logging.getLogger().setLevel(logging.ERROR)

    os.chdir(r"C:\Users\cgokmen\research\scene_graphs")

    tasks = []
    for fn in tqdm.tqdm(list(glob.glob("*_replay.hdf5"))[:1]):
        try:
            real_fn = os.path.join("H:\My Drive\BEHAVIOR resources\Virtual Reality Demos", fn.replace("_replay", ""))
            f = h5py.File(real_fn, "r")

            # print(fn)
            task = f.attrs["/metadata/atus_activity"]
            task_id = f.attrs["/metadata/activity_definition"]
            scene = f.attrs["/metadata/scene_id"]
            instance_id = f.attrs["/metadata/instance_id"]
            urdf_file = f.attrs["/metadata/urdf_file"]

            config = {}
            config["task"] = task
            config["task_id"] = task_id
            config["scene"] = "igibson"
            config["robot"] = "BehaviorRobot"
            config["scene_id"] = scene
            config["instance_id"] = instance_id
            config["urdf_file"] = urdf_file
            config["online_sampling"] = False
            config["should_activate_behavior_robot"] = False

            env = iGibsonEnv(
                config_file=config,
            )
            task = env.task
            tasks.append(task)
        except ValueError as e:
            print("Something went wrong with ", fn, ":", e)
        finally:
            f.close()

    # Convert the task to a graph
    t = tasks[0]
    G = nx.DiGraph()
    h1 = t.goal_conditions[0]
    h1.to_graph(G)


if __name__ == "__main__":
    main()
