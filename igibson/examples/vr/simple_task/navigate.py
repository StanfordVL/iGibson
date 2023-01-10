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
n_obstacle_layers = 3
horizontal_obstacles = 5
vertical_obstacles = 0
obstacle_vel = 0.5

n_vertical_obstacles = n_obstacle_layers * vertical_obstacles
n_horizontal_obstacles = n_obstacle_layers * horizontal_obstacles


initial_x, initial_z = -1.3, 0.75
default_robot_pose = ([initial_x, 0, 1], [0, 0, 0, 1])
duck_position = [1.4, 0, 1.5]
distance = duck_position[0] - initial_x
intro_paragraph = """Welcome to the navigate experiment!
There will be a duck at the end of the hallway. 
----------------------------------------------------------------
1. Move forward and push the duck in front of you.
2. Stay within the corridor while you are moving.
2. Try to go straight and avoid obstacles along the way!
----------------------------------------------------------------
Go to the starting point (red marker) and face the duck
Press menu button on the right controller to begin.
"""


def import_obj(s):
    # hallway setup
    for i in range(2):
        wall_left = ArticulatedObject(
            "igibson/examples/vr/visual_disease_demo_mtls/plane/wall.urdf", scale=0.07, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        )
        s.import_object(wall_left)
        wall_left.set_position_orientation([i * 2 - 0.6, 1, 1], [0.707, 0, 0, 0.707])
        wall_right = ArticulatedObject(
            "igibson/examples/vr/visual_disease_demo_mtls/plane/wall.urdf", scale=0.07, rendering_params={"use_pbr": False, "use_pbr_mapping": False}
        )
        s.import_object(wall_right)
        wall_right.set_position_orientation([i * 2 - 0.6, -1, 1], [0.707, 0, 0, 0.707])

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
    duck.set_position_orientation(duck_position, [0.5, 0.5, 0.5, 0.5])

    ret = {}
    ret["obstacles"] = obstacles
    ret["duck"] = duck
    return ret



def set_obj_pos(objs):
    # object setup
    objs["duck"].set_position_orientation(duck_position, [0.5, 0.5, 0.5, 0.5])
    objs["duck"].force_wakeup()

    for i in range(n_vertical_obstacles):
        objs["obstacles"][i].set_position_orientation([(i % n_obstacle_layers + 1) * distance / (n_obstacle_layers + 1) + initial_x, random() * 1.8 - 0.9, random() * 1.5], [0, 0, 0, 1])
        objs["obstacles"][i].set_velocities([([0, 0, (-1) ** randint(0, 1) * obstacle_vel], [0, 0, 0])])
        objs["obstacles"][i].force_wakeup()
    for i in range(n_horizontal_obstacles):
        idx = i + n_vertical_obstacles
        objs["obstacles"][idx].set_position_orientation([(i % n_obstacle_layers + 1) * distance / (n_obstacle_layers + 1) + initial_x, random() * 2 - 1, i // n_obstacle_layers * 0.25 + initial_z], [0, 0, 0, 1])
        objs["obstacles"][idx].set_velocities([([0, (-1) ** randint(0, 1) * obstacle_vel, 0], [0, 0, 0])])
        objs["obstacles"][idx].force_wakeup()


def main(s, log_writer, disable_save, debug, robot, objs, args): 
    is_valid, success = True, False 
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
            if (obj_pos > 2 and obj_vel > 0) or (obj_pos < 0.4 and obj_vel < 0):
                objs["obstacles"][i].set_velocities([([0, 0, -obj_vel], [0, 0, 0])])
        for i in range(n_horizontal_obstacles):
            idx = i + n_vertical_obstacles
            obj_pos, obj_vel = objs["obstacles"][idx].get_position()[1], objs["obstacles"][idx].get_velocities()[0][0][1]
            if (obj_pos > 1 and obj_vel > 0) or (obj_pos < -1 and obj_vel < 0):
                objs["obstacles"][idx].set_velocities([([0, -obj_vel, 0], [0, 0, 0])])

        if np.linalg.norm(objs["duck"].get_position() - duck_position) > 0.1:
            if success_time:
                if time.time() - success_time > 0.5:
                    success = True
                    break
            else:
                success_time = time.time()

    return is_valid, success

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main() 