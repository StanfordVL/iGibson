import cv2
import logging
import numpy as np
import os 
import pdb
import random
import time 

import bddl
from IPython import embed

from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
# from igibson.activity.tests.task_scene_to_view import TASK_SCENE_TO_VIEW


bddl.set_backend("iGibson")

task_choices = [
    "packing_lunches_filtered",
    "assembling_gift_baskets_filtered",
    "organizing_school_stuff_filtered",
    "re-shelving_library_books_filtered",
    "serving_hors_d_oeuvres_filtered",
    "putting_away_toys_filtered",
    "putting_away_Christmas_decorations_filtered",
    "putting_dishes_away_after_cleaning_filtered",
    "cleaning_out_drawers_filtered",
]
Rs_int_tasks = [
    "storing_food",
    "sorting_mail",
    "polishing_silver",
    "polishing_furniture",
    "cleaning_sneakers",
    "cleaning_oven",
    "cleaning_up_the_kitchen_only",
    "cleaning_windows",
    "cleaning_microwave_oven",
    "bottling_fruit",
    "assembling_gift_baskets",
    "re-shelving_library_books",
    "sorting_books",
    "putting_away_Christmas_decorations",
    "filling_an_Easter_basket",
    "putting_away_Halloween_decorations",
    "cleaning_out_drawers",
    "cleaning_kitchen_cupboard",
    "chopping_vegetables",
    "filling_a_Christmas_stocking",
    "locking_every_window",
    "preserving_food"
]
Rs_int_tasks_kitchen = [
    "storing_food",
    # "sorting_mail",
    # "polishing_silver",
    # "polishing_furniture",
    # "cleaning_sneakers",
    "cleaning_oven",
    "cleaning_up_the_kitchen_only",
    "cleaning_windows",
    "cleaning_microwave_oven",
    "bottling_fruit",
    "assembling_gift_baskets",
    "re-shelving_library_books",
    "sorting_books",
    "putting_away_Christmas_decorations",
    "filling_an_Easter_basket",
    "putting_away_Halloween_decorations",
    "cleaning_out_drawers",
    "cleaning_kitchen_cupboard",
    "chopping_vegetables",
    "filling_a_Christmas_stocking",
    # "locking_every_window",
    "preserving_food"
]
# task = "assembling_gift_baskets"
# task = "putting_away_Christmas_decorations"
# task = "chopping_vegetables"
# task = "re-shelving_library_books"
# task = "storing_food"
# task = "sorting_books"
# task = "polishing_silver"
# task = "sorting_mail"
# task = "cleaning_up_the_kitchen_only"
task = "packing_lunches"
task_id = 0
# scene_id = "Rs_int"
scene_id = "Wainscott_0_int"
# scene_id = "Pomaria_1_int"
num_init = 0

igbhvr_act_inst = iGBEHAVIORActivityInstance(task, activity_definition=task_id)
scene_kwargs = {
    # 'load_object_categories': ['oven', 'fridge', 'countertop', 'cherry', 'sausage', 'tray'],
    "not_load_object_categories": ["ceilings"],
    "urdf_file": "{}_task_{}_{}_{}".format(scene_id, task, task_id, num_init), # *****
}
# try:
#     view_idx = 0
#     viewer_settings = TASK_SCENE_TO_VIEW[(task, scene_id)][view_idx]
# except KeyError:
#     logging.warning('Using default viewer settings')
#     viewer_settings = {}
#     pdb.set_trace()
simulator = Simulator(mode="iggui", image_width=960, image_height=720)#, viewer_settings=viewer_settings)

init_success = igbhvr_act_inst.initialize_simulator(
    scene_id=scene_id,
    simulator=simulator,
    load_clutter=True,
    should_debug_sampling=False,
    scene_kwargs=scene_kwargs,
    online_sampling=False,
)
print("success")
embed()
# while not simulator.viewer.exit:
# # for i in range(100):
#     igbhvr_act_inst.simulator.step()
# filename = os.path.join("/home/frieda/Documents/code/iGibson/screenshots", "{}_{}_{}_{}_{}.png".format(scene_id, task, task_id, num_init, view_idx))
# frame = cv2.cvtColor(np.concatenate(simulator.renderer.render(modes=("rgb")), axis=1), cv2.COLOR_RGB2BGR)
# print(frame)
# print(frame.max())
# if frame.max():
#     cv2.imwrite(filename, (frame * 255).astype(np.uint8))
#     print(filename)
#     print(os.path.isfile(filename))

while True:
    igbhvr_act_inst.simulator.step()
    success, sorted_conditions = igbhvr_act_inst.check_success()
    # print('.' if not success else '!', end='')
    print("TASK SUCCESS:", success)
    if not success:
        print("FAILED CONDITIONS:", sorted_conditions["unsatisfied"])
        embed()
    else:
        pass
