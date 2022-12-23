import logging
import random
import time
import numpy as np
from igibson.objects.articulated_object import ArticulatedObject


# Hyper parameters
horizontal_level = 6
vertical_level = 3
sampling_ratio = 0.7
total_num_objects_before_sampling = horizontal_level ** 2 * vertical_level
total_num_objects = int(total_num_objects_before_sampling * sampling_ratio)

num_of_duck = 1
horizontal_offset = 0.6
vertical_offset = 0.5
initial_x, initial_y, initial_z = -1.5, -1.5, 1
default_robot_pose = ([0, 0, 0], [0, 0, 0, 1])
intro_paragraph = """   Welcome to the navigate experiment!
    There will be a yellow duck among a bunch of spheres. Navigate using the touchpad and push the duck using your hand.
    Press menu button on the right controller to proceed."""

def import_obj(s):
    # obstacles and ducks setup
    obstacles = []
    ducks = []

    for i in range(total_num_objects - num_of_duck):
        obstacles.append(ArticulatedObject(
            "igibson/examples/vr/visual_disease_demo_mtls/sphere.urdf", scale=40, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        ))
        s.import_object(obstacles[-1])
        obstacles[-1].set_position([0, 0, i]) # dummy position
    for _ in range(num_of_duck):
        ducks.append(ArticulatedObject(
            "duck_vhacd.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        ))
        s.import_object(ducks[-1])
        ducks[-1].set_position([0, 1, i]) # dummy position

    ret = {}
    ret["obstacles"] = obstacles
    ret["ducks"] = ducks
    return ret



def set_obj_pos(objs):
    # object setup
    duck_pos = 1        
    sampled_objects = random.sample(range(total_num_objects_before_sampling), total_num_objects)
    duck_index = random.sample(sampled_objects, num_of_duck)
    for idx in duck_index:
        sampled_objects.remove(idx)
    duck_pos = []

    for i, duck_idx in enumerate(duck_index):
        x, y, z = duck_idx % horizontal_level, duck_idx // horizontal_level % horizontal_level, duck_idx // horizontal_level // horizontal_level % vertical_level
        d_pos = [initial_x + x * horizontal_offset, initial_y + y * horizontal_offset, initial_z + z * vertical_offset]
        duck_pos.append(d_pos)
        objs["ducks"][i].set_position_orientation(d_pos, [0.5, 0.5, 0.5, 0.5])
        objs["ducks"][i].force_wakeup()
    for i, obj_idx in enumerate(sampled_objects):
        x, y, z = obj_idx % horizontal_level, obj_idx // horizontal_level % horizontal_level, obj_idx // horizontal_level // horizontal_level % vertical_level
        objs["obstacles"][i].set_position_orientation([initial_x + x * horizontal_offset, initial_y + y * horizontal_offset, initial_z + z * vertical_offset], [0, 0, 0, 1])
        objs["obstacles"][i].force_wakeup()

    ret = {}
    ret["duck_pos"] = duck_pos
    return ret



def main(s, log_writer, disable_save, debug, robot, objs, args): 
    success, terminate = False, False 
    success_time = 0  
    done = set()
    # Main simulation loop
    while True:
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()       
        robot.apply_action(s.gen_vr_robot_action())
        if debug:
            s.update_vi_effect()

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            terminate = True
            break

        # Start counting time by pressing overlay toggle
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break
        
        ducks_checked = 0
        for i in range(num_of_duck):
            if i not in done and np.linalg.norm(objs["ducks"][ducks_checked].get_position() - args["duck_pos"][i]) > 0.1:
                objs["ducks"][ducks_checked].set_position([0, 0, -i])
                done.add(i)
            ducks_checked += 1

        if len(done) == num_of_duck:
            if success_time:
                if time.time() - success_time > 1:
                    success = True
                    break
            else:
                success_time = time.time()

    return success, terminate

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 