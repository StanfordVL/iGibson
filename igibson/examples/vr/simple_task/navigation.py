import logging
import random
import numpy as np
from igibson.objects.articulated_object import ArticulatedObject


num_of_duck = 1
initial_x, initial_y = -5, -5
default_robot_pose = ([0, -6, 1], [0, 0, 0, 1])

def import_obj(s):
    # obstacles and ducks setup
    obstacles = []
    ducks = []
    for _ in range(100 - num_of_duck):
        obstacles.append(ArticulatedObject(
            "sphere_1cm.urdf", scale=50, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        ))
        s.import_object(obstacles[-1])
    for _ in range(num_of_duck):
        ducks.append(ArticulatedObject(
            "duck_vhacd.urdf", scale=2, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        ))
        s.import_object(ducks[-1])

    ret = {}
    ret["obstacles"] = obstacles
    ret["ducks"] = ducks
    return ret



def set_obj_pos(objs):
    # object setup
    duck_pos = random.sample(range(50, 100), num_of_duck)
    heights = [random.random() * 0.5 + 1.2 for _ in range(100)]
    ducks_positioned = 0
    for i in range(100):
        if i in duck_pos:
            objs["ducks"][ducks_positioned].set_position_orientation([initial_x + i % 10, initial_y + i // 10, heights[i]], [0.5, 0.5, 0.5, 0.5])
            ducks_positioned += 1
        else:
            objs["obstacles"][i - ducks_positioned].set_position_orientation([initial_x + i % 10, initial_y + i // 10, heights[i]], [0.5, 0.5, 0.5, 0.5])

    ret = {}
    ret["heights"] = heights
    ret["duck_pos"] = duck_pos
    return ret



def main(s, log_writer, disable_save, robot, objs, args): 
    success, terminate = False, False   
    done = set()
    start_time = 0
    # Main simulation loop
    while True:
        s.step()
        if log_writer and not disable_save:
            log_writer.process_frame()       
        robot.apply_action(s.gen_vr_robot_action())
        s.update_post_processing_effect()

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            terminate = True
            break

        # Start counting time by pressing overlay toggle
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break

        # update post processing
        ducks_checked = 0
        for i in range(100):
            if i in args["duck_pos"]:
                if i not in done and np.linalg.norm(objs["ducks"][ducks_checked].get_position() - np.array([initial_x + i % 10, initial_y + i // 10, args["heights"][i]])) > 0.1:
                    objs["ducks"][ducks_checked].set_position([0, 0, -i])
                    done.add(i)
                ducks_checked += 1
            else:
                objs["obstacles"][i - ducks_checked].set_position_orientation([initial_x + i % 10, initial_y + i // 10, args["heights"][i]], [0.5, 0.5, 0.5, 0.5])

        if len(done) == num_of_duck:
            success = True
            break

    return success, terminate

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 