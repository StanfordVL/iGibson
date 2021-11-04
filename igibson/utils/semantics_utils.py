import os
import yaml

import igibson
from igibson.utils.constants import SemanticClass


def get_class_name_to_class_id(starting_class_id=SemanticClass.SCENE_OBJS + 1):
    """
    Get mapping from semantic class name to class id

    :param starting_class_id: starting class id for scene objects
    """
    category_txt = os.path.join(igibson.ig_dataset_path, "metadata/categories.txt")
    class_name_to_class_id = dict()
    if os.path.isfile(category_txt):
        with open(category_txt) as f:
            for line in f.readlines():
                class_name_to_class_id[line.strip()] = starting_class_id
                starting_class_id += 1
    class_name_to_class_id["agent"] = SemanticClass.ROBOTS
    return class_name_to_class_id

if __name__ == "__main__":
    class_mapping_dict = get_class_name_to_class_id()
    classes, ids = zip(*sorted(class_mapping_dict.items(), key=lambda x: x[1]))
    n_categories = len(class_mapping_dict)
    classes = ['placeholder1', 'placeholder2', 'placeholder3', 'agent'] + list(classes)[1:]
    ids = [0, 1, 2, 3] + list(ids)[1:]

    with open('igibson_dataset.yaml', 'w') as file:
        yaml.dump({'nc': len(classes), 'names': classes }, file)
