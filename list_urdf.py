import json
import os
import bddl
import igibson
import math

import pandas as pd


def main():
    scene_json = os.path.join(os.path.dirname(bddl.__file__), "../utils", "activity_to_preselected_scenes.json")
    activity_conditions = os.path.join(os.path.dirname(bddl.__file__), "activity_conditions")
    urdf_paths = []
    urdf_files = []
    scene_ids = []
    tasks = []
    task_ids = []
    init_ids = []
    start_initialization = 0
    num_initializations = 30
    task_id = 0

    with open(scene_json) as f:
        activity_to_scenes = json.load(f)

    for task in activity_to_scenes:
        scene_choices = activity_to_scenes[task]

        if task not in os.listdir(activity_conditions):
            continue

        for scene_id in scene_choices:
            for init_id in range(start_initialization, start_initialization + num_initializations):
                init_ids.append(init_id)
                task_ids.append(task_id)
                tasks.append(task)
                scene_ids.append(scene_id)
                urdf_path = "{}_task_{}_{}_{}".format(scene_id, task, task_id, init_id)
                urdf_filepath = os.path.join("scenes", scene_id, "urdf", urdf_path + ".urdf")
                urdf_test_path = os.path.join(igibson.ig_dataset_path, "scenes", scene_id, "urdf", urdf_path + ".urdf")
                if not os.path.exists(urdf_test_path):
                    import pdb

                    pdb.set_trace()
                urdf_paths.append(urdf_path)
                urdf_files.append(urdf_filepath)

    df = pd.DataFrame(
        {
            "urdf_filepath": urdf_files,
            "urdf_path": urdf_paths,
            "scene_id": scene_ids,
            "task": tasks,
            "task_ids": task_ids,
            "init_ids": init_ids,
        }
    )
    df.drop_duplicates(inplace=True)
    df.reset_index(inplace=True)
    df.to_csv("urdf_manifest.csv")

    size = math.ceil(df.shape[0] / 8)
    list_of_dfs = [df.loc[i : i + size - 1, :] for i in range(0, len(df), size)]
    for idx, df in enumerate(list_of_dfs):
        df.to_csv("urdf_manifest_split_{}.csv".format(idx))


if __name__ == "__main__":
    main()
