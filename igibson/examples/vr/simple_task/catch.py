import logging
import time
import random
from igibson.objects.articulated_object import ArticulatedObject


# Hyper parameters
num_trials = {
    "training": 1,
    "collecting": 1
}
total_trial_per_round = 10
default_robot_pose = ([0, 0, 1], [0, 0, 0, 1])
intro_paragraph = """ Welcome to the catch experiment!
In this experiment you can see a ball flying towards you. 
----------------------------------------------------------------
1. Catch the ball by pressing the trigger.
2. KEEP HOLDING the ball until the ball disappears.
3. Try to use your dominate hand for catching.
----------------------------------------------------------------
Go to the starting point (red marker) and face the wall
Press menu button on the right controller to begin.
"""


def import_obj(s):
    ret = {"ball": ArticulatedObject("sphere_small.urdf", scale=1.5, rendering_params={"use_pbr": False, "use_pbr_mapping": False})}
    s.import_object(ret["ball"])
    return ret

def set_obj_pos(objs):
    pass


def main(s, log_writer, disable_save, debug, robot, objs, args):
    is_valid = True
    start_time = time.time()
    cur_time = start_time
    episode_len = 4
    is_bounced = False
    gamma = 0.85
    init_x_pos = 8

    rand_z = random.random() * 0.5 + 2.25
    rand_y = random.random() * 0.5 - 0.25
    objs["ball"].set_position((init_x_pos, rand_y, rand_z))
    objs["ball"].set_velocities([([-6, 0, 4], [0, 0, 0])])
    objs["ball"].force_wakeup()

    trial_offset = 1
    total_trial = 0
    success_trial = 0
    total_consecutive_success = 0

    # Main simulation loop
    while True:
        robot.apply_action(s.gen_vr_robot_action())
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()    
           
        s.update_vi_effect(debug)

        ball_pos = objs["ball"].get_position()

        cur_time = time.time()
        if (cur_time - start_time > episode_len):
            if trial_offset:
                trial_offset -= 1
            elif ball_pos[2] > 0.25:
                total_trial += 1
                success_trial += 1
                total_consecutive_success += 1
            else:
                total_consecutive_success = 0
                total_trial += 1
            
            if args["training"] and total_consecutive_success == 5:
                break
            elif total_trial == total_trial_per_round:
                break
            start_time = cur_time
            rand_z = random.random() * 0.5 + 2.25
            rand_y = random.random() * 0.5 - 0.25
            objs["ball"].set_position((init_x_pos, rand_y, rand_z))
            objs["ball"].set_velocities([([-6, 0, 4], [0, 0, 0])])
            is_bounced = False
            objs["ball"].force_wakeup()
            continue

        if (ball_pos[2] < 0.07 and not is_bounced):
            is_bounced = True
            objs["ball"].set_velocities([([-4 * gamma, 0, (2 * 9.8 * rand_z) ** 0.5 * gamma], [0, 0, 0])])

        # End demo by pressing overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            is_valid = False
            break
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break
    
    sr = success_trial / total_trial if total_trial else 0
    print(f"Total: {total_trial}, Success: {success_trial}, SR: {sr}")
    return is_valid, sr

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()