import itertools
import json
import os

from tasknet.object_taxonomy import ObjectTaxonomy

import gibson2
from pynput import keyboard
import numpy as np
import pybullet as p

from gibson2.objects.articulated_object import URDFObject
from gibson2.objects.visual_marker import VisualMarker
from gibson2.scenes.empty_scene import EmptyScene
from gibson2.simulator import Simulator
from gibson2.utils.assets_utils import download_assets

download_assets()

ABILITY_NAME = "cleaningTool"
CATEGORIES = [
    "broom",
    "carpet_sweeper",
    "scraper",
    "scrub_brush",
    "toothbrush",
    "vacuum",
]
USE_ABILITY_TO_FETCH_CATEGORIES = False

OBJECT_TAXONOMY = ObjectTaxonomy()


def get_categories():
    dir = os.path.join(gibson2.ig_dataset_path, 'objects')
    return [cat for cat in os.listdir(dir) if os.path.isdir(get_category_directory(cat))]


def get_category_directory(category):
    return os.path.join(gibson2.ig_dataset_path, 'objects', category)


def get_obj(folder, **kwargs):
    return URDFObject(os.path.join(folder, os.path.basename(folder) + ".urdf"), name="obj", model_path=folder, **kwargs)


def get_metadata_filename(objdir):
    return os.path.join(objdir, "misc", "metadata.json")


def main():
    # Collect the relevant categories.
    categories = CATEGORIES
    if USE_ABILITY_TO_FETCH_CATEGORIES:
        categories = []
        for cat in get_categories():
            # Check that the category has this label.
            klass = OBJECT_TAXONOMY.get_class_name_from_igibson_category(cat)
            if not klass:
                continue

            if not OBJECT_TAXONOMY.has_ability(klass, ABILITY_NAME):
                continue

            categories.append(cat)

    print("%d categories: %s" % (len(categories), ", ".join(categories)))

    # Now collect the actual objects.
    objects = []
    objects_by_category = {}
    for cat in categories:
        cd = get_category_directory(cat)
        objects_by_category[cat] = []
        for objdir in os.listdir(cd):
            objdirfull = os.path.join(cd, objdir)
            objects.append(objdirfull)
            objects_by_category[cat].append(objdirfull)

    print("%d objects.\n" % len(objects))

    max_bbox = [0.9, 0.9, 0.9]
    avg = {"size": max_bbox, "density": 67.0}
    current_x = 0

    s = Simulator(mode='gui')
    scene = EmptyScene()
    s.import_scene(scene)
    for cat in categories:
        cd = get_category_directory(cat)
        for objdir in os.listdir(cd):
            objdirfull = os.path.join(cd, objdir)

            obj = get_obj(objdirfull, fit_avg_dim_volume=True, avg_obj_dims=avg, flags=p.URDF_ENABLE_SLEEPING)
            # obj = get_obj(objdirfull, bounding_box=max_bbox, flags=p.URDF_ENABLE_SLEEPING)
            s.import_object(obj)
            obj_pos = np.array([current_x, 0., 0.5])
            obj.set_position(obj_pos)
            x_sign = -1 if current_x < 0 else 1
            current_x = (abs(current_x) + 2) * x_sign * -1

    try:
        while True:
            pass #s.step()
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
