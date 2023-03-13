import logging
import os
import platform

import numpy as np
import pybullet as p
import yaml

import igibson
from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from igibson.action_primitives.baseline_action_primitives import BaselineActionPrimitives
from igibson.objects.articulated_object import URDFObject
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_model_path
from igibson.utils.utils import parse_config


def execute_controller(ctrl_gen, robot, s):
    for action in ctrl_gen:
        robot.apply_action(action)
        s.step()

def grasp_object(s, robot, obj, controller: BaselineActionPrimitives):
    try:
        """Grasp the obj that's on the floor of the room."""
        print("Trying to GRASP %s" % (obj.name))
        obj = s.scene.objects_by_category[obj.name][0]
        execute_controller(controller.grasp(obj), robot, s)
        print("GRASP the %s succeeded!" % (obj.name))
        return True
    except ActionPrimitiveError:
        return False


def main(selection="user", headless=False, short_exec=False, num_trials=5):
    """
    Launches a simulator scene and showcases a variety of semantic action primitives such as navigation, grasping,
    placing, opening and closing and their success rates across multiple objects
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Create the simulator.
    s = Simulator(
        mode="headless" if headless else "gui_non_interactive" if platform.system() != "Darwin" else "gui_interactive",
        image_width=512,
        image_height=512,
        device_idx=0,
        use_pb_gui=(not headless and platform.system() != "Darwin"),
    )
    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=["floors", "coffee_table"]#, "bottom_cabinet", "door", "sink", "coffee_table", "fridge"]
    )
    s.import_scene(scene)

    # Create a custom tray object for the grasping test.
    model_path = get_ig_model_path("tray", "tray_000")
    model_filename = os.path.join(model_path, "tray_000.urdf")
    avg_category_spec = get_ig_avg_category_specs()
    tray = URDFObject(
        filename=model_filename,
        category="tray",
        name="tray",
        avg_obj_dims=avg_category_spec.get("tray"),
        fit_avg_dim_volume=True,
        model_path=model_path,
    )
    s.import_object(tray)
    tray.set_position_orientation([0, 1, 0.3], p.getQuaternionFromEuler([0, np.pi / 2, 0]))

    # Load the robot and place it in the scene.
    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_mp_behavior_task.yaml"))
    config["robot"]["show_visual_head"] = True
#    config["robot"]["grasping_mode"] = "sticky" 
    robot = BehaviorRobot(**config["robot"], grasping_mode="sticky")
    s.import_robot(robot)
    robot.set_position_orientation([0, 0, 1], [0, 0, 0, 1])
    robot.apply_action(
        np.zeros(
            robot.action_dim,
        )
    )

    # Run some steps to let physics settle.
    for _ in range(300):
        s.step()

    # Create an Action Primitive Set and use it to convert high-level actions to low-level actions and execute.
    controller = BaselineActionPrimitives(None, scene, robot)


    successes = 0
    try:
        # The pick-and-place demo is always run.
        for i in range(num_trials):
            print("running trial %d" % i)
            successes += grasp_object(s, robot, tray, controller)

        print("%f success rate" % (successes/num_trials))
        print("Done with trials")
        # If we're not running in headless mode, let the simulator run idle after we are done to allow user to inspect.
        if not headless:
            while True:
                action = np.zeros(robot.action_dim)
                robot.apply_action(action)
                s.step()
    finally:
        s.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
