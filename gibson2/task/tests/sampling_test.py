from gibson2.task.task_base import iGTNTask

import tasknet
tasknet.set_backend("iGibson")


igtn_task = iGTNTask(
    'putting_dishes_away_after_cleaning_filtered', task_instance=0)
scene_kwargs = {
    # 'load_object_categories': ['coffee_table', 'breakfast_table', 'countertop', 'fridge', 'table_lamp', 'sofa', 'bottom_cabinet', 'bottom_cabinet_no_top', 'top_cabinet'],
    'not_load_object_categories': ['ceilings'],
}
init_success = igtn_task.initialize_simulator(
    scene_id='Rs_int',
    mode='gui',
    load_clutter=False,
    should_debug_sampling=True,
    scene_kwargs=scene_kwargs
)
assert init_success

while True:
    igtn_task.simulator.step()
    success, sorted_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', sorted_conditions['unsatisfied'])
    else:
        pass
