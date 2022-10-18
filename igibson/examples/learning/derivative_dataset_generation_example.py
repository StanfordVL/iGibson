import logging
import os
import sys

import yaml

import igibson
from igibson.utils import assets_utils
from igibson.utils.derivative_dataset.pipeline import DerivativeDatasetPipeline


def main():
    # Get slurm job prefix info.
    job_id = int(os.getenv("SLURM_JOBID", 0))
    array_id = int(os.getenv("SLURM_ARRAY_TASK_ID", 0))
    task_id = int(os.getenv("SLURM_LOCALID", 0))
    prefix = f"{job_id}-{array_id}-{task_id}"

    # Automatically infer scene ID from slurm info.
    available_scenes = assets_utils.get_available_ig_scenes()
    scene_id = available_scenes[array_id % len(available_scenes)]

    # Get config file.
    config_name = sys.argv[1]
    config_filename = os.path.join(igibson.configs_path, "derivative_dataset", config_name)
    with open(config_filename, "r") as f:
        config = yaml.load(f, yaml.Loader)

    # Start pipeline.
    DerivativeDatasetPipeline(scene_id=scene_id, prefix=prefix, **config).generate()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
