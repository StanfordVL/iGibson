import gibson2
import os
from gibson2.utils.constants import SemanticClass

def get_class_name_to_class_id(starting_class_id=SemanticClass.SCENE_OBJS):
    category_txt = os.path.join(gibson2.ig_dataset_path,
                                'metadata/categories.txt')
    class_name_to_class_id = dict()
    if os.path.isfile(category_txt):
        with open(category_txt) as f:
            for line in f.readlines():
                class_name_to_class_id[line.strip()] = starting_class_id
                starting_class_id += 1
    return class_name_to_class_id
