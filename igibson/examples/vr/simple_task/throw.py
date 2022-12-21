import time
import logging
import igibson
from random import random
from igibson.object_states import OnTop
from igibson.objects.articulated_object import ArticulatedObject


default_robot_pose = ([-0.75, -1, 0.7], [0, 0, 0, 1])
intro_paragraph = """   Welcome to the throw experiment!
    There will be a basket on the ground and a ball on the table. Grab the ball using your hand and throw it into the basket.
    Press menu button on the right controller to proceed."""

def import_obj(s):
    ret = {}
    ret["table"] = ArticulatedObject("table/table.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["sphere"] = ArticulatedObject("sphere_small.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["basket"] = ArticulatedObject(f"{igibson.ig_dataset_path}/objects/basket/e3bae8da192ab3d4a17ae19fa77775ff/e3bae8da192ab3d4a17ae19fa77775ff.urdf", scale=2, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    for obj in ret.values():
        s.import_object(obj)
    return ret



def set_obj_pos(objs):
    # objects
    objs["table"].set_position_orientation((-1.2, -1.75, 0), (0, 0, 0, 1))
    objs["sphere"].set_position_orientation((-0.6, -1.4, 0.66), (0.000000, 0.707107, 0.000000, 0.707107))

    basket_y = random()  - 1.5
    objs["basket"].set_position((0.7, basket_y, 0.2))
    objs["basket"].set_orientation((0, 0, 0, 1))
    for obj in objs:
        objs[obj].force_wakeup()

    return {"basket_y": basket_y}



def main(s, log_writer, disable_save, debug, robot, objs, ret):
    success, terminate = False, False
    complete_time = 0
    # Main simulation loop
    while True:
        s.step()
        if log_writer and not disable_save:
            log_writer.process_frame()     
        robot.apply_action(s.gen_vr_robot_action())
        if debug:
            s.update_vi_effect()

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            terminate = True
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break

        # sphere in basket: task complete
        if objs["sphere"].states[OnTop].get_value(objs["basket"], use_ray_casting_method=True):
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
