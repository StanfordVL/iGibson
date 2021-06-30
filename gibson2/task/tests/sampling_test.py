from gibson2.task.task_base import iGTNTask
from IPython import embed
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.simulator import Simulator

import tasknet
tasknet.set_backend("iGibson")

activity = 'assembling_gift_baskets'
scene_id = 'Rs_int'

igtn_task = iGTNTask(
    activity, task_instance=0)
scene_kwargs = {
    'not_load_object_categories': ['ceilings'],
}
settings = MeshRendererSettings(texture_scale=1)
simulator = Simulator(mode='headless',
                      image_width=960,
                      image_height=720,
                      rendering_settings=settings)
init_success = igtn_task.initialize_simulator(
    scene_id=scene_id,
    simulator=simulator,
    load_clutter=False,
    should_debug_sampling=True,
    scene_kwargs=scene_kwargs
)
assert init_success
print('success')
embed()

while True:
    igtn_task.simulator.step()
    success, sorted_conditions = igtn_task.check_success()
    print('TASK SUCCESS:', success)
    if not success:
        print('FAILED CONDITIONS:', sorted_conditions['unsatisfied'])
    else:
        pass
