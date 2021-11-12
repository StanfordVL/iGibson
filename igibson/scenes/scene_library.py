import os

SCENE_SOURCE = ["IG", "CUBICASA", "THREEDFRONT"]

from igibson.utils.assets_utils import (
    get_3dfront_scene_path,
    get_cubicasa_scene_path,
    get_ig_scene_path,
)

def get_scene_path(scene_name, scene_type, object_randomization=None, object_randomization_idx=None):
    if object_randomization:
        if object_randomization_idx is None:
            filename = scene_name
        else:
            filename = "{}_random_{}".format(scene_name, object_randomization_idx)
    else:
        filename = "{}_best".format(scene_name)

    if scene_type not in SCENE_SOURCE:
        raise ValueError("Unsupported scene source: {}".format(scene_type))
    if scene_type == "IG":
        scene_dir = get_ig_scene_path(scene_name)
    elif scene_type == "CUBICASA":
        scene_dir = get_cubicasa_scene_path(scene_name)
    else:
        scene_dir = get_3dfront_scene_path(scene_name)

    return os.path.join(scene_dir, "urdf", "{}.urdf".format(filename)), scene_dir

