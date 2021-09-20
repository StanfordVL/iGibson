import argparse
import os
import subprocess

import bddl

# loading_the_dishwasher, making_tea: need to remove goal condition sampling (or potentially use ray-casting based)
# polishing_silver: need to change particle sampling AABB to 0.01

skip_tasks = ["loading_the_dishwasher", "making_tea", "polishing_silver"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_trials", type=int, default=100, help="Maximum number of trials to try sampling.")
    parser.add_argument(
        "--num_initializations", type=int, default=1, help="Number of initialization per PDDL per scene."
    )
    parser.add_argument("--start_initialization", type=int, default=0, help="Starting idx for initialization")
    parser.add_argument(
        "--object_randomization",
        type=int,
        default=0,
        help="Whether to enable furniture object randomization (0 is False, 1 is True)",
    )

    args = parser.parse_args()

    condition_dir = os.path.join(os.path.dirname(bddl.__file__), "activity_conditions")
    for task in sorted(os.listdir(condition_dir)):
        if task in skip_tasks:
            continue
        task_dir = os.path.join(condition_dir, task)
        if os.path.isdir(task_dir):
            for task_id_file in sorted(os.listdir(task_dir)):
                task_id = task_id_file.replace("problem", "")[0]
                if task_id != "0":
                    continue
                subprocess.call(
                    "python sampling_saver.py --task {} --task_id {} --max_trials {} --num_initializations {} --start_initialization {} --object_randomization {}".format(
                        task,
                        task_id,
                        args.max_trials,
                        args.num_initializations,
                        args.start_initialization,
                        args.object_randomization,
                    ),
                    shell=True,
                )


if __name__ == "__main__":
    main()
