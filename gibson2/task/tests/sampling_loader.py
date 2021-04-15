from gibson2.task.task_base import iGTNTask
from IPython import embed

import tasknet
tasknet.set_backend("iGibson")

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
task = 're-shelving_library_books_filtered'
task_id = 0
scene = 'Rs_int'

igtn_task = iGTNTask(task, task_instance=task_id)
scene_kwargs = {
    # 'load_object_categories': ['oven', 'fridge', 'countertop', 'cherry', 'sausage', 'tray'],
    'not_load_object_categories': ['ceilings'],
    'urdf_file': '{}_task_{}_{}_0'.format(scene, task, task_id),
}
init_success = igtn_task.initialize_simulator(
    scene_id=scene,
    mode='gui',
    load_clutter=True,
    should_debug_sampling=False,
    scene_kwargs=scene_kwargs,
    online_sampling=False,
    offline_sampling=True,
)

print('loading done')
embed()

while True:
    igtn_task.simulator.step()
    success, sorted_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', sorted_conditions['unsatisfied'])
    else:
        pass
