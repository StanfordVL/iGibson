import bddl

from igibson.activity.activity_base import iGBEHAVIORActivityInstance

bddl.set_backend("iGibson")


igbhvr_act_inst = iGBEHAVIORActivityInstance("sampling_test", activity_definition=4)
igbhvr_act_inst.initialize_simulator(scene_id="Rs_int", mode="gui", load_clutter=True)

while True:
    igbhvr_act_inst.simulator.step()
    success, sorted_conditions = igbhvr_act_inst.check_success()
    print("ACTIVITY SUCCESS:", success)
    if not success:
        print("FAILED CONDITIONS:", sorted_conditions["unsatisfied"])
    else:
        pass
