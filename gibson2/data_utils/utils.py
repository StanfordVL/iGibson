import os
import json
import gibson2


def load_json(file_path):
    with open(file_path) as f:
        obj = json.load(f)
    return obj

def get_train_models():
    return load_json(os.path.join(os.path.dirname(gibson2.__file__), 'data', 'train.json'))
