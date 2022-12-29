import logging
from random import random, randint
import time
import numpy as np
from igibson.objects.articulated_object import ArticulatedObject


# Hyper parameters
num_trials = {
    "training": 2,
    "collecting": 5
}
n_obstacle_layers = 2
horizontal_obstacles = 2
vertical_obstacles = 2
obstacle_vel = 0.7

n_vertical_obstacles = n_obstacle_layers * vertical_obstacles
n_horizontal_obstacles = n_obstacle_layers * horizontal_obstacles


initial_x, initial_z = -0.6, 1
default_robot_pose = ([-1.6, 0, 1], [0, 0, 0, 1])
duck_positions = [[1.4, 0, 1.5], [-1.4, 0, 1.5]]
intro_paragraph = """Welcome to the navigate experiment!
There will be 2 yellow ducks at each end of the hallway. 
----------------------------------------------------------------
1. Approach and push the duck in front of you.
2. Turn around to face the opposite direction
3. Approach and push the other duck in front of you.
4. Try to avoid obstacles along the way!
----------------------------------------------------------------
Go to the starting point (red marker) and face the duck
Press menu button on the right controller to begin.
"""


def import_obj(s):
    # obstacles and ducks setup
    obstacles = []
    for i in range(n_vertical_obstacles + n_horizontal_obstacles):
        obstacles.append(ArticulatedObject(
            "igibson/examples/vr/visual_disease_demo_mtls/sphere.urdf", scale=15, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        ))
        s.import_object(obstacles[-1])
        obstacles[-1].set_position([20, 20, i]) # dummy position

    duck = ArticulatedObject("duck_vhacd.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    s.import_object(duck)
    duck.set_position_orientation(duck_positions[0], [0.5, 0.5, 0.5, 0.5])

    ret = {}
    ret["obstacles"] = obstacles
    ret["duck"] = duck
    return ret



def set_obj_pos(objs):
    # object setup
    objs["duck"].set_position_orientation(duck_positions[0], [0.5, 0.5, 0.5, 0.5])
    objs["duck"].force_wakeup()

    for i in range(n_vertical_obstacles):
        objs["obstacles"][i].set_position_orientation([i % n_obstacle_layers * 1.4 + initial_x, random() * 1.5 - 0.75, i // n_obstacle_layers * 0.3 + initial_z], [0, 0, 0, 1])
        objs["obstacles"][i].set_velocities([([0, 0, (-1) ** randint(0, 1) * obstacle_vel], [0, 0, 0])])
        objs["obstacles"][i].force_wakeup()
    for i in range(n_horizontal_obstacles):
        idx = i + n_vertical_obstacles
        objs["obstacles"][idx].set_position_orientation([i % n_obstacle_layers * 1.4 + initial_x, random() * 1.5 - 0.75, i // n_obstacle_layers * 0.3 + initial_z], [0, 0, 0, 1])
        objs["obstacles"][idx].set_velocities([([0, (-1) ** randint(0, 1) * obstacle_vel, 0], [0, 0, 0])])
        objs["obstacles"][idx].force_wakeup()


def main(s, log_writer, disable_save, debug, robot, objs, args): 
    is_valid, success = True, False 
    duck_pos = duck_positions[0]
    success_time = 0
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

        # Start counting time by pressing overlay toggle
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break

        for i in range(n_vertical_obstacles):
            obj_pos, obj_vel = objs["obstacles"][i].get_position()[2], objs["obstacles"][i].get_velocities()[0][0][2]
            if (obj_pos > 2 and obj_vel > 0) or (obj_pos < 0.2 and obj_vel < 0):
                objs["obstacles"][i].set_velocities([([0, 0, -obj_vel], [0, 0, 0])])
        for i in range(n_horizontal_obstacles):
            idx = i + n_vertical_obstacles
            obj_pos, obj_vel = objs["obstacles"][idx].get_position()[1], objs["obstacles"][idx].get_velocities()[0][0][1]
            if (obj_pos > 1 and obj_vel > 0) or (obj_pos < -1 and obj_vel < 0):
                objs["obstacles"][idx].set_velocities([([0, -obj_vel, 0], [0, 0, 0])])

        if np.linalg.norm(objs["duck"].get_position() - duck_pos) > 0.1:
            if duck_pos == duck_positions[0]:
                duck_pos = duck_positions[1]
                objs["duck"].set_position_orientation(duck_pos, [0.5, 0.5, 0.5, 0.5])
                objs["duck"].set_velocities([([0, 0, 0], [0, 0, 0])])
                continue
            elif success_time:
                if time.time() - success_time > 0.5:
                    success = True
                    break
            else:
                success_time = time.time()

    return is_valid, success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 