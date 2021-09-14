import cv2
import logging
logging.getLogger().setLevel(logging.DEBUG)
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
from igibson.activity.tests.task_scene_to_view import TASK_SCENE_TO_VIEW


bddl.set_backend("iGibson")
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
task = "storing_food"
task_id = 0
scene_id = "Rs_int"
# scene_id = "Wainscott_0_int"
# scene_id = "Pomaria_1_int"
num_init = 0
filename = os.path.join("/home/frieda/Documents/code/iGibson/screenshots", "{}_{}_{}_{}.png".format(scene_id, task, task_id, num_init))
while os.path.isfile(filename):
    num_init += 1
    filename = os.path.join("/home/frieda/Documents/code/iGibson/screenshots", "{}_{}_{}_{}.png".format(scene_id, task, task_id, num_init))

igbhvr_act_inst = iGBEHAVIORActivityInstance(task, activity_definition=task_id)
scene_kwargs = {
    # 'load_object_categories': ['oven', 'fridge', 'countertop', 'cherry', 'sausage', 'tray'],
    "not_load_object_categories": ["ceilings"],
    "urdf_file": "{}_task_{}_{}_{}".format(scene_id, task, task_id, num_init), # *****
}
simulator = Simulator(mode="iggui", image_width=960, image_height=720, viewer_settings={})

init_success = igbhvr_act_inst.initialize_simulator(
    scene_id=scene_id,
    simulator=simulator,
    load_clutter=True,
    should_debug_sampling=False,
    scene_kwargs=scene_kwargs,
    online_sampling=False,
)
print("success")
# # embed()
time.sleep(15)
for view_idx in range(len(TASK_SCENE_TO_VIEW[(task, scene_id)])):
    igbhvr_act_inst.simulator.step()
    viewer_settings = TASK_SCENE_TO_VIEW[(task, scene_id)][view_idx]
    camera_pose = viewer_settings['initial_pos']
    view_direction = viewer_settings['initial_view_direction']
    up = viewer_settings['initial_up']
    simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, up)
    igbhvr_act_inst.simulator.step()
    frame = cv2.cvtColor(np.concatenate(simulator.renderer.render(modes=("rgb")), axis=1), cv2.COLOR_RGB2BGR)
    os.makedirs("/home/frieda/Documents/code/iGibson/screenshots", exist_ok=True)
    filename = os.path.join("/home/frieda/Documents/code/iGibson/screenshots", "{}_{}_{}_{}_{}.png".format(scene_id, task, task_id, num_init, view_idx))
    # print(frame)
    # print(frame.max())
    if frame.max():
        cv2.imwrite(filename, (frame * 255).astype(np.uint8))
        print(filename)
        print(os.path.isfile(filename))
    else:
        logging.warning('Frame is all zeros')
        simulator.disconnect() # .renderer.release()
        pdb.set_trace()
    time.sleep(1)

# *****
for i in range(num_init+1,10):
    num_init = i
    filename = os.path.join("/home/frieda/Documents/code/iGibson/screenshots", "{}_{}_{}_{}_{}.png".format(scene_id, task, task_id, num_init, view_idx))
    if os.path.isfile(filename):
        logging.warning(f'Already snapshotted {filename}')
        continue
    scene_kwargs = {
        # 'load_object_categories': ['oven', 'fridge', 'countertop', 'cherry', 'sausage', 'tray'],
        "not_load_object_categories": ["ceilings"],
        "urdf_file": "{}_task_{}_{}_{}".format(scene_id, task, task_id, num_init), # *****
    }
    try:
        igbhvr_act_inst.scene = InteractiveIndoorScene(scene_id, **scene_kwargs)
        # simulator.import_ig_scene(scene)
        igbhvr_act_inst.import_scene()
        print(f"success{num_init}")
        # embed()
        time.sleep(15)
        for view_idx in range(len(TASK_SCENE_TO_VIEW[(task, scene_id)])):
            igbhvr_act_inst.simulator.step()
            viewer_settings = TASK_SCENE_TO_VIEW[(task, scene_id)][view_idx]
            camera_pose = viewer_settings['initial_pos']
            view_direction = viewer_settings['initial_view_direction']
            up = viewer_settings['initial_up']
            simulator.renderer.set_camera(camera_pose, camera_pose + view_direction, up)
            igbhvr_act_inst.simulator.step()
            frame = cv2.cvtColor(np.concatenate(simulator.renderer.render(modes=("rgb")), axis=1), cv2.COLOR_RGB2BGR)
            # print(frame)
            # print(frame.max())
            if frame.max():
                cv2.imwrite(filename, (frame * 255).astype(np.uint8))
                print(filename)
                print(os.path.isfile(filename))
            else:
                logging.warning('Frame is all zeros')
                simulator.disconnect() # .renderer.release()
                pdb.set_trace()
            time.sleep(1)
    except Exception as e:
        print(e)
