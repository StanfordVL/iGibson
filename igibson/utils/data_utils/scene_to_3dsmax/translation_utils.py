import os

import yaml

import igibson
from igibson.utils import assets_utils


def _load_translations():
    with open(os.path.join(igibson.ig_dataset_path, "metadata", "model_rename.yaml"), "r") as f:
        old_to_new = yaml.load(f)

    new_to_old = {v: k for k, v in old_to_new.items()}

    return old_to_new, new_to_old


_OLD_TO_NEW, _NEW_TO_OLD = _load_translations()


def old_to_new(old_cat, old_model):
    return _OLD_TO_NEW[f"{old_cat}/{old_model}"].split("/")


def new_to_old(new_cat, new_model):
    return _NEW_TO_OLD[f"{new_cat}/{new_model}"].split("/")


def model_to_pair(path):
    if not os.path.isdir(path):
        path = os.path.split(path)[0]

    path, model = os.path.split(path)
    _, category = os.path.split(path)

    assert category in assets_utils.get_all_object_categories(), "Invalid category %s" % category

    return category, model
