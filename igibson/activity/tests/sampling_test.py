import bddl
from IPython import embed

from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.simulator import Simulator

bddl.set_backend("iGibson")

activity = "assembling_gift_baskets"
scene_id = "Rs_int"

igbhvr_act_inst = iGBEHAVIORActivityInstance(activity, activity_definition=0)
scene_kwargs = {
    "not_load_object_categories": ["ceilings"],
}
settings = MeshRendererSettings(texture_scale=1)
simulator = Simulator(mode="headless", image_width=960, image_height=720, rendering_settings=settings)
init_success = igbhvr_act_inst.initialize_simulator(
    scene_id=scene_id, simulator=simulator, load_clutter=False, should_debug_sampling=True, scene_kwargs=scene_kwargs
)
assert init_success
print("success")
embed()

while True:
    igbhvr_act_inst.simulator.step()
    success, sorted_conditions = igbhvr_act_inst.check_success()
    print("TASK SUCCESS:", success)
    if not success:
        print("FAILED CONDITIONS:", sorted_conditions["unsatisfied"])
    else:
        pass
