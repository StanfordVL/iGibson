import logging
import time
import random
from igibson.objects.articulated_object import ArticulatedObject
from igibson.object_states import OnTop

# Hyper parameters
num_trials = {
    "training": 2,
    "collecting": 2
}
num_of_placing_obj = 4
timeout = 90
default_robot_pose = ([0, 0, 1], [0, 0, 0, 1])
object_scale_factor = [0.8, 1.2]
target_height = [1.0425, 1.0453, 1.0497, 1.0539]
intro_paragraph = """Welcome to the place experiment!
There will be 4 baskets and 4 cubes on the desk.
They have 1 to 1 correspondence based on their shapes and sizes. 
----------------------------------------------------------------
1. Place the cubes COMPLETELY INSIDE the baskets! 
2. Try to use your dominant hand.
3. Move your hand away from the table when restarting.
----------------------------------------------------------------
Go to the starting point (red marker) and face the desk
Press menu button on the right controller to begin.
"""


def import_obj(s):
    # table as static object
    table = ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/table/table.urdf", scale=1, rendering_params={"use_pbr": False, "use_pbr_mapping": False})
    s.import_object(table)
    table.set_position((0.8000000, 0.00000, 0.000000))
    table.set_orientation((0.00000, 0.000000, 0.707107, 0.707107))
    # basket and cube
    basket = []
    cube = []
    for i in range(num_of_placing_obj // 2):
        basket.append(ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/basket/square_basket.urdf", scale=0.3 * object_scale_factor[i]))
        s.import_object(basket[-1])
        basket[-1].set_position([20, 20, i])
        basket.append(ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/basket/rectangle_basket.urdf", scale=0.3 * object_scale_factor[i]))
        s.import_object(basket[-1])
        basket[-1].set_position([20, 20, 2 + i])
        cube.append(ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/cube/square_cube.urdf", scale=object_scale_factor[i]))
        s.import_object(cube[-1])
        cube[-1].set_position([20, 20, 4 + i])
        cube.append(ArticulatedObject("igibson/examples/vr/visual_disease_demo_mtls/cube/rectangle_cube.urdf", scale=object_scale_factor[i]))
        s.import_object(cube[-1])
        cube[-1].set_position([20, 20, 6 + i])
    ret = {}
    ret["basket"] = basket
    ret["cube"] = cube
    return ret



def set_obj_pos(objs):
    # object setup
    basket_pos = random.sample(range(10), num_of_placing_obj)
    for i in range(num_of_placing_obj):
        objs["basket"][i].set_position([random.random() * 0.5 + 0.4, basket_pos[i] / 10 - 0.5, 1.035])
        objs["basket"][i].set_orientation([0, 0, 0, 1])
        objs["basket"][i].set_velocities([([0, 0, 0], [0, 0, 0])])
        objs["basket"][i].force_wakeup()

        objs["cube"][i].set_position([random.random() * 0.5 + 0.4, random.random()* 0.2 - i * 0.15, 1.1])
        objs["cube"][i].set_orientation([0, 0, 0, 1])
        objs["cube"][i].set_velocities([([0, 0, 0], [0, 0, 0])])
        objs["cube"][i].force_wakeup()


def main(s, log_writer, disable_save, debug, robot, objs, args):
    is_valid, success = True, False
    success_time = 0
    start_time = time.time()
    # Main simulation loop
    while True:
        robot.apply_action(s.gen_vr_robot_action())
        s.step(print_stats=debug)
        if log_writer and not disable_save:
            log_writer.process_frame()     
        s.update_vi_effect(debug)

        # End demo by pressing left overlay toggle
        if s.query_vr_event("left_controller", "overlay_toggle"):
            is_valid = False
            break

        # refresh demo by pressing right overlay toggle
        if s.query_vr_event("right_controller", "overlay_toggle"):
            break
        completed = 0
        for i in range(num_of_placing_obj):
            cube_z = objs["cube"][i].get_position()[2] 
            if objs["cube"][i].states[OnTop].get_value(objs["basket"][i], use_ray_casting_method=True) and (cube_z < 1.029 or abs(cube_z - target_height[i]) < 5e-3):
                completed += 1
        if completed == num_of_placing_obj:
            if success_time:
                if time.time() - success_time > 0.5:
                    success = True
                    break
            else:
                success_time = time.time()
        else:
            success_time = 0

        # timeout
        if time.time() - start_time > timeout:
            break
    return is_valid, success
    


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()