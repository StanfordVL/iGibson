import logging
import os
import time
import random
from igibson.objects.articulated_object import ArticulatedObject
from igibson.object_states import Inside

num_of_placing_obj = 4
default_robot_pose = ([0, 0, 1.5], [0, 0, 0, 1])
intro_paragraph = """   Welcome to the place experiment!
    There will be 4 baskets and 4 cubes on the desk. Place the cubes COMPLETELY INSIDE the baskets (order doesn't matter)
    Press menu button on the right controller to proceed."""

def import_obj(s):
    # table as static object
    table = ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/table/table.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    s.import_object(table)
    table.set_position((0.8000000, 0.00000, 0.000000))
    table.set_orientation((0.00000, 0.000000, 0.707107, 0.707107))
    # basket and cube
    basket = []
    cube = []
    for _ in range(num_of_placing_obj):
        basket.append(ArticulatedObject(os.path.join("igibson/examples/vr/visual_disease_demo_mtls/basket/e3bae8da192ab3d4a17ae19fa77775ff.urdf"), scale=0.3))
        s.import_object(basket[-1])
        cube.append(ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/butter/butter_000.urdf", scale=0.05))
        s.import_object(cube[-1])
    ret = {}
    ret["basket"] = basket
    ret["cube"] = cube
    return ret



def set_obj_pos(objs):
    # object setup
    basket_pos = random.sample(range(10), num_of_placing_obj)
    for i in range(num_of_placing_obj):
        objs["basket"][i].set_position([random.random() * 0.5 + 0.4, basket_pos[i] / 10 - 0.5, 1.05])
        objs["basket"][i].set_orientation([0, 0, 0, 1])
        objs["basket"][i].force_wakeup()

        objs["cube"][i].set_position([random.random() * 0.5 + 0.4, random.random()* 0.2 - i * 0.15, 1.1])
        objs["cube"][i].set_orientation([random.random(), random.random(), random.random(), random.random()])
        objs["cube"][i].force_wakeup()

def main(s, log_writer, disable_save, debug, robot, objs, ret):
    success, terminate = False, False
    success_time = 0
    # Main simulation loop
    while True:
        robot.apply_action(s.gen_vr_robot_action())
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()     
        s.update_vi_effect(debug)

        # End demo by pressing left overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            terminate = True
            break

        # refresh demo by pressing right overlay toggle
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break

        completed = 0
        for b in objs["basket"]:
            for c in objs["cube"]:
                if c.states[Inside].get_value(b, use_ray_casting_method=True) and c.get_position()[2] < 1.0475:
                    completed += 1
                    break
        if completed == num_of_placing_obj:
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