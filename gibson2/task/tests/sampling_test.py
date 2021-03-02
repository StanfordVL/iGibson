from gibson2.task.task_base import iGTNTask

import tasknet
tasknet.set_backend("iGibson")


igtn_task = iGTNTask('sampling_test', task_instance=6)
scene_kwargs = {
    'load_object_categories': ['coffee_table', 'breakfast_table', 'countertop', 'fridge', 'table_lamp', 'sofa'],
    'not_load_object_categories': ['ceilings'],
}
igtn_task.initialize_simulator(
    scene_id='Rs_int',
    mode='gui',
    load_clutter=False,
    should_debug_sampling=True,
    scene_kwargs=scene_kwargs
)

while True:
    igtn_task.simulator.step()
    success, sorted_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', sorted_conditions['unsatisfied'])
    else:
        pass
