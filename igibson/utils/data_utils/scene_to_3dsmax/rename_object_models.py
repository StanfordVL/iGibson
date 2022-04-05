import os
import random
import string

import yaml

import igibson
from igibson.utils import assets_utils

N = 6


def rand_str():
    return "".join(random.choices(string.ascii_lowercase, k=N))


def main():
    map = {}
    existing_hashes = set()

    categories = assets_utils.get_all_object_categories()
    for cat in categories:
        for mod in assets_utils.get_object_models_of_category(cat):
            hash = rand_str()
            assert hash not in existing_hashes

            orig_name = f"{cat}/{mod}"
            new_name = f"{cat}/{hash}"

            map[orig_name] = new_name
            existing_hashes.add(hash)

    with open(os.path.join(igibson.ig_dataset_path, "metadata", "model_rename.yaml"), "w") as f:
        yaml.dump(map, f)

    print("Translation map created successfully.")


if __name__ == "__main__":
    main()
