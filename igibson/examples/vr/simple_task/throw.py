import time
import logging
import igibson
from random import shuffle
import numpy as np
from itertools import product
from igibson.object_states import Inside
from igibson.objects.articulated_object import ArticulatedObject


# Hyper parameters
num_trials = {
    "training": 15,
    "collecting": 9
}
default_robot_pose = ([0, 0, 1], [0, 0, 0, 1])
basket_pos_choices = list(product([1.6, 1.8, 2.0], [-0.5, 0, 0.5]))
intro_paragraph = """Welcome to the throw experiment!
There will be a basket on the ground and a ball on the table.
--------------------------------
1. Grab the ball and throw it into the basket.
2. Do NOT move or lean your body forward!
3. Try to use your dominant hand when throwing.
--------------------------------
Go to the starting point (red marker) and face the basket
Press menu button on the right controller to begin.
"""


def import_obj(s):
    ret = {}
    ret["table"] = ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/table/table.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["sphere"] = ArticulatedObject("sphere_small.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    ret["basket"] = ArticulatedObject(f"{igibson.ig_dataset_path}/objects/basket/e3bae8da192ab3d4a17ae19fa77775ff/e3bae8da192ab3d4a17ae19fa77775ff.urdf", scale=1.5, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    for obj in ret.values():
        s.import_object(obj)
    ret["table"].set_position_orientation((-0.45, -0.75, 0), (0, 0, 0, 1))
    # dummy positions
    ret["basket"].set_position_orientation((2, 0, 0.1), (0, 0, 0, 1))
    ret["sphere"].set_position_orientation((-0.45, -0.75, 0.1), (0, 0, 0, 1))   
    # get basket pos choices
    shuffle(basket_pos_choices)
    ret["basket_pos_choices"] = basket_pos_choices
    ret["basket_pos_choices_idx"] = 0

    return ret



def set_obj_pos(objs):
    # objects
    objs["sphere"].set_position_orientation((0.15, -0.4, 1.1), (0, 0, 0, 1))
    basket_pos_choice = objs["basket_pos_choices"][objs["basket_pos_choices_idx"]]
    objs["basket_pos_choices_idx"] = (objs["basket_pos_choices_idx"] + 1) % 9
    objs["basket"].set_position((basket_pos_choice[0], basket_pos_choice[1], 0.15))
    objs["basket"].set_orientation((0, 0, 0, 1))
    objs["sphere"].force_wakeup()
    objs["basket"].force_wakeup()


def main(s, log_writer, disable_save, debug, robot, objs, args):
    is_valid, success = True, False
    complete_time = None
    # Main simulation loop
    while True:
        robot.apply_action(s.gen_vr_robot_action())
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()     
        s.update_vi_effect(debug)


        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            is_valid = False
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break

        # sphere in basket: task complete
        if objs["sphere"].states[Inside].get_value(objs["basket"], use_ray_casting_method=True):
            if complete_time:
                if time.time() - complete_time > 0.5:
                    success = True
                    break
            else:
                complete_time = time.time()
        # sphere fell to the ground but not in basket: task failed
        elif objs["sphere"].get_position()[2] < 0.05:
            if complete_time:
                if time.time() - complete_time > 0.5:
                    success = False
                    break
            else:
                complete_time = time.time()
        else:
            complete_time = 0

    return is_valid, success


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
