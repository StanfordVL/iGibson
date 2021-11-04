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
    class_name_to_id = get_class_name_to_class_id()
    class_id_to_name = {int(y): x for x, y in class_name_to_id.items()}
    classes = [
        class_id_to_name[x] if x in class_id_to_name else "placeholder" + str(x)
        for x in range(max(class_id_to_name) + 1)
    ]

    with open("igibson_dataset.yaml", "w") as file:
        yaml.dump({"nc": len(classes), "names": classes}, file)
