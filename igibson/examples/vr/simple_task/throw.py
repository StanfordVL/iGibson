import time
import logging
import igibson
from random import random
from igibson.object_states import Inside
from igibson.objects.articulated_object import ArticulatedObject


default_robot_pose = ([0, 0, 1.5], [0, 0, 0, 1])
intro_paragraph = """   Welcome to the throw experiment!
    There will be a basket on the ground and a ball on the table. Grab the ball using your hand and throw it into the basket.
    Press menu button on the right controller to proceed."""

def import_obj(s):
    ret = {}
    ret["table"] = ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/table/table.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["sphere"] = ArticulatedObject("sphere_small.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["basket"] = ArticulatedObject(f"{igibson.ig_dataset_path}/objects/basket/e3bae8da192ab3d4a17ae19fa77775ff/e3bae8da192ab3d4a17ae19fa77775ff.urdf", scale=2, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    for obj in ret.values():
        s.import_object(obj)
    return ret



def set_obj_pos(objs):
    # objects
    objs["table"].set_position_orientation((-0.45, -0.75, 0), (0, 0, 0, 1))
    objs["sphere"].set_position_orientation((0.15, -0.4, 1.1), (0, 0, 0, 1))

    basket_y = random() - 0.5
    objs["basket"].set_position((1.75, basket_y, 0.15))
    objs["basket"].set_orientation((0, 0, 0, 1))
    for obj in objs:
        objs[obj].force_wakeup()

    return {"basket_y": basket_y}



def main(s, log_writer, disable_save, debug, robot, objs, ret):
    success, terminate = False, False
    complete_time = 0
    # Main simulation loop
    while True:
        robot.apply_action(s.gen_vr_robot_action())
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()     
        s.update_vi_effect(debug)


        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            terminate = True
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break

        # sphere in basket: task complete
        if objs["sphere"].states[Inside].get_value(objs["basket"], use_ray_casting_method=True):
            if complete_time:
                if time.time() - complete_time > 1:
                    success = True
                    break
            else:
                complete_time = time.time()
        # sphere fell to the ground but not in basket: task failed
        elif objs["sphere"].get_position()[2] < 0.05:
            if complete_time:
                if time.time() - complete_time > 1:
                    success = False
                    break
            else:
                complete_time = time.time()
        else:
            complete_time = 0

    return success, terminate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
