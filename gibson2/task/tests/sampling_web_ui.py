from gibson2.task.task_base import iGTNTask
from IPython import embed
import pybullet as p

import tasknet
tasknet.set_backend("iGibson")


igtn_task = iGTNTask(
    'sampling_test', task_instance=7)
scene_kwargs = {
    'load_object_categories': ['coffee_table', 'breakfast_table'],
    # 'load_object_categories': ['coffee_table', 'breakfast_table', 'countertop', 'fridge', 'table_lamp', 'sofa', 'bottom_cabinet', 'bottom_cabinet_no_top', 'top_cabinet'],
    'not_load_object_categories': ['ceilings'],
}
igtn_task.initialize_simulator(
    scene_id='Rs_int',
    mode='gui',
    load_clutter=False,
    should_debug_sampling=False,
    scene_kwargs=scene_kwargs,
    online_sampling=False,
    offline_sampling=False,
)
state_id = p.saveState()

while True:
    accept_scene = igtn_task.check_scene()
    assert accept_scene
    accept_scene = igtn_task.sample()
    assert accept_scene
    print('sample')
    embed()
    for sim_obj in igtn_task.newly_added_objects:
        igtn_task.scene.remove_object(sim_obj)
        for id in sim_obj.body_ids:
            p.removeBody(id)
    p.restoreState(state_id)

print('init done')
embed()

while True:
    igtn_task.simulator.step()
    success, sorted_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', sorted_conditions['unsatisfied'])
    else:
        pass
