from gibson2.task.task_base import iGTNTask
from IPython import embed

import tasknet
tasknet.set_backend("iGibson")


igtn_task = iGTNTask('re-shelving_library_books_filtered', task_instance=0)
scene_kwargs = {
    # 'load_object_categories': ['coffee_table', 'breakfast_table', 'countertop', 'fridge', 'table_lamp', 'sofa', 'bottom_cabinet', 'bottom_cabinet_no_top', 'top_cabinet'],
    'not_load_object_categories': ['ceilings'],
    'urdf_file': 'Rs_int_re-shelving_library_books_filtered_0_0'
}
igtn_task.initialize_simulator(
    scene_id='Rs_int',
    mode='gui',
    load_clutter=True,
    should_debug_sampling=False,
    scene_kwargs=scene_kwargs,
    online_sampling=False,
)
# sim_obj_to_pddl_obj = {
#     value.name: {'object_scope': key} for key, value in igtn_task.object_scope.items()}
# print('initialize_simulator')
# igtn_task.scene.save_modified_urdf('test', sim_obj_to_pddl_obj)
print('saving done')
embed()

while True:
    igtn_task.simulator.step()
    success, sorted_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', sorted_conditions['unsatisfied'])
    else:
        pass
