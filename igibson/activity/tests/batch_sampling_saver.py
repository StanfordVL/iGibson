import argparse
import json
import os
import subprocess
from shutil import copyfile

import bddl

import igibson

# loading_the_dishwasher, making_tea: need to remove goal condition sampling (or potentially use ray-casting based)
# polishing_silver: need to change particle sampling AABB to 0.01

skip_tasks = ["loading_the_dishwasher", "making_tea", "polishing_silver"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_trials", type=int, default=100, help="Maximum number of trials to try sampling.")
    args = parser.parse_args()

    condition_dir = os.path.join(os.path.dirname(bddl.__file__), "activity_definitions")
    for task in sorted(os.listdir(condition_dir)):
        if task != "assembling_gift_baskets":
            continue
        if task in skip_tasks:
            continue
        task_dir = os.path.join(condition_dir, task)
        if not os.path.isdir(task_dir):
            continue
        for task_id_file in sorted(os.listdir(task_dir)):
            task_id = task_id_file.replace("problem", "")[0]
            if task_id != "0":
                continue
            subprocess.call(
                "python sampling_saver.py --task {} --task_id {} --max_trials {} --num_initializations {} --start_initialization {} --object_randomization {}".format(
                    task,
                    task_id,
                    args.max_trials,
                    1,
                    100,
                    0,
                ),
                shell=True,
            )
            subprocess.call(
                "python sampling_saver.py --task {} --task_id {} --max_trials {} --num_initializations {} --start_initialization {} --object_randomization {}".format(
                    task,
                    task_id,
                    args.max_trials,
                    1,
                    101,
                    1,
                ),
                shell=True,
            )

    scene_json = os.path.join(os.path.dirname(bddl.__file__), "../utils", "activity_to_preselected_scenes.json")

    with open(scene_json) as f:
        activity_to_scenes = json.load(f)

    src_urdf_paths = set()
    init_ids = [100, 101]
    for task in sorted(os.listdir(condition_dir)):
        if task != "assembling_gift_baskets":
            continue
        if task in skip_tasks:
            continue
        if task not in activity_to_scenes:
            continue
        task_dir = os.path.join(condition_dir, task)
        if not os.path.isdir(task_dir):
            continue
        scene_choices = activity_to_scenes[task]
        for task_id_file in sorted(os.listdir(task_dir)):
            task_id = task_id_file.replace("problem", "")[0]
            if task_id != "0":
                continue

            for scene_id in scene_choices:
                for init_id in init_ids:
                    urdf_path = "{}_task_{}_{}_{}".format(scene_id, task, task_id, init_id)
                    full_path = os.path.join(igibson.ig_dataset_path, "scenes", scene_id, "urdf", urdf_path + ".urdf")
                    src_urdf_paths.add(full_path)

    existing_src_urdf_paths = [full_path for full_path in src_urdf_paths if os.path.isfile(full_path)]
    non_existing_src_urdf_paths = [full_path for full_path in src_urdf_paths if not os.path.isfile(full_path)]
    if len(non_existing_src_urdf_paths) == 0:
        scene_caches_folder = "/scene_caches"
        os.makedirs(scene_caches_folder, exist_ok=True)
        for full_path in existing_src_urdf_paths:
            dst_path = os.path.join(scene_caches_folder, os.path.basename(full_path))
            copyfile(full_path, dst_path)
        print("Done.")
    else:
        for full_path in non_existing_src_urdf_paths:
            print(full_path)
        print("Missing scene caches are the above. Please re-run.")


if __name__ == "__main__":
    main()
