"""This file contains utils for BEHAVIOR demo replay checkpoints."""
import json
import os

import pybullet as p

def save_checkpoint(simulator, root_directory):
    bullet_path = os.path.join(root_directory, "%d.bullet" % simulator.frame_count)
    json_path = os.path.join(root_directory, "%d.json" % simulator.frame_count)

    p.saveBullet(bullet_path)
    state_dump = {}
    for name, obj in simulator.scene.objects_by_name.items():
        state_dump[name] = obj.dump_state()
    with open(json_path, "w") as f:
        json.dump(state_dump, f)


def load_checkpoint(simulator, root_directory, frame):
    bullet_path = os.path.join(root_directory, "%d.bullet" % frame)
    json_path = os.path.join(root_directory, "%d.json" % frame)

    p.restoreState(fileName=bullet_path)
    with open(json_path, "r") as f:
        state_dump = json.load(f)
        for name, obj in simulator.scene.objects_by_name.items():
            obj.load_state(state_dump[name])