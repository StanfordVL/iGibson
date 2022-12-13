import time
import logging
import igibson
from random import random
from igibson.object_states import OnTop
from igibson.objects.articulated_object import ArticulatedObject


default_robot_pose = ([-0.75, -1, 0.7], [0, 0, 0, 1])
intro_paragraph = "Welcome to the throw experiment! In this experiment there will be a basket and a ball on the table. We need to grab the ball and throw it into the basket."

def import_obj(s):
    ret = {}
    ret["table1"] = ArticulatedObject("table/table.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["table2"] = ArticulatedObject("table/table.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["sphere"] = ArticulatedObject("sphere_small.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["basket"] = ArticulatedObject(f"{igibson.ig_dataset_path}/objects/basket/e3bae8da192ab3d4a17ae19fa77775ff/e3bae8da192ab3d4a17ae19fa77775ff.urdf", scale=2, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    for obj in ret.values():
        s.import_object(obj)
    return ret



def set_obj_pos(objs):
    # objects
    objs["table1"].set_position_orientation((1.000000, -1.000000, -0.20000), (0.000000, 0.000000, 0.707107, 0.707107))
    objs["table2"].set_position_orientation((-0.050000, -1.000000, 0.00000), (0.000000, 0.000000, 0.707107, 0.707107))
    objs["sphere"].set_position_orientation((-0.400000, -1.00000, 0.750000), (0.000000, 0.707107, 0.000000, 0.707107))

    basket_y = random()  - 1.5
    objs["basket"].set_position((0.7, basket_y, 0.62))
    objs["basket"].set_orientation((0, 0, 0, 1))
    objs["basket"].force_wakeup()

    return {"basket_y": basket_y}



def main(s, log_writer, disable_save, debug, robot, objs, ret):
    success, terminate = False, False
    success_time = 0
    # Main simulation loop
    while True:
        s.step()
        if log_writer and not disable_save:
            log_writer.process_frame()     
        robot.apply_action(s.gen_vr_robot_action())
        if debug:
            s.update_vi_effect()

        # keep basket still
        objs["basket"].set_position((0.7, ret["basket_y"], 0.62))
        objs["basket"].set_orientation((0, 0, 0, 1))

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            terminate = True
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break

        if objs["sphere"].states[OnTop].get_value(objs["basket"], use_ray_casting_method=True):
            if success_time:
                if time.time() - success_time > 1:
                    success = True
                    break
            else:
                success_time = time.time()
        else:
            success_time = 0

    return success, terminate


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
