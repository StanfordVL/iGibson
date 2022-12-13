import logging
import os
import time
import random
import igibson
from igibson.objects.articulated_object import ArticulatedObject

total_trial_per_round = 10
default_robot_pose = ([-2.75, 0.85, 0.7], [0, 0, 0, 1])
intro_paragraph = "Welcome to the catch experiment! In this experiment we can see a ball flying towards us. We need to catch the ball by moving our hand to the ball."

def import_obj(s):
    ret = {"ball": ArticulatedObject(os.path.join(igibson.ig_dataset_path, "objects/ball/ball_000/ball_000.urdf"), scale=0.16)}
    s.import_object(ret["ball"])
    return ret

def set_obj_pos(objs):
    pass

def main(s, log_writer, disable_save, debug, robot, objs, ret):
    terminate = False
    start_time = time.time()
    cur_time = start_time
    episode_len = 4
    is_bounced = False
    gamma = 0.9
    init_y_pos = -7

    rand_z = random.random() * 0.5 + 2
    objs["ball"].set_position((-3, init_y_pos , rand_z))
    objs["ball"].set_velocities([([0, 6, 4], [0, 0, 0])])
    objs["ball"].force_wakeup()

    trial_offset = 5
    total_trial = 0
    success_trial = 0

    # Main simulation loop
    while True:
        s.step()
        if log_writer and not disable_save:
            log_writer.process_frame()       
        robot.apply_action(s.gen_vr_robot_action())
        if debug:
            s.update_vi_effect()

        ball_pos = objs["ball"].get_position()

        cur_time = time.time()
        if (cur_time - start_time > episode_len):
            if trial_offset:
                trial_offset -= 1
            elif ball_pos[2] > 0.25:
                total_trial += 1
                success_trial += 1
            else:
                total_trial += 1

            if total_trial == total_trial_per_round:
                break
            start_time = cur_time
            rand_z = random.random() * 0.5 + 2 
            objs["ball"].set_position((-3, init_y_pos , rand_z))
            objs["ball"].set_velocities([([0, 6, 4], [0, 0, 0])])
            is_bounced = False
            objs["ball"].force_wakeup()
            continue


        if (ball_pos[2] < 0.07 and not is_bounced):
            is_bounced = True
            objs["ball"].set_velocities([([0, 2 * gamma, (2 * 9.8 * rand_z) ** 0.5 * gamma], [0, 0, 0])])

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            terminate = True
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break
    
    sr = success_trial / total_trial if total_trial else 0
    print(f"Total: {total_trial}, Success: {success_trial}, SR: {sr}")
    return sr, terminate

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()