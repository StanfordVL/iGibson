import os

import igibson
from igibson.utils.constants import MAX_CLASS_COUNT, SemanticClass


def get_class_name_to_class_id():
    """
    Get mapping from semantic class name to class id

    :param starting_class_id: starting class id for scene objects
    """
    existing_classes = {item.value for item in SemanticClass}
    category_txt = os.path.join(igibson.ig_dataset_path, "metadata/categories.txt")
    class_name_to_class_id = {"agent": SemanticClass.ROBOTS}  # Agents should have the robot semantic class.
    starting_class_id = 0
    if os.path.isfile(category_txt):
        with open(category_txt) as f:
            for line in f.readlines():
                while starting_class_id in existing_classes:
                    starting_class_id += 1
                assert starting_class_id < MAX_CLASS_COUNT, "Class ID overflow: MAX_CLASS_COUNT is {}.".format(
                    MAX_CLASS_COUNT
                )
                class_name_to_class_id[line.strip()] = starting_class_id
                starting_class_id += 1

    return class_name_to_class_id


CLASS_NAME_TO_CLASS_ID = get_class_name_to_class_id()
