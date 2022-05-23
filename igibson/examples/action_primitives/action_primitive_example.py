import logging
import os
import platform

import numpy as np
import pybullet as p
import yaml

import igibson
from igibson.action_primitives.action_primitive_set_base import ActionPrimitiveError
from igibson.action_primitives.starter_semantic_action_primitives import StarterSemanticActionPrimitives
from igibson.envs.igibson_env import iGibsonEnv
from igibson.objects.articulated_object import URDFObject
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_model_path
from igibson.utils.utils import parse_config

ATTEMPTS = 5


def execute_controller(ctrl_gen, robot, s):
    for action in ctrl_gen:
        robot.apply_action(action)
        s.step()


def go_to_sink_and_toggle(s, robot, controller: StarterSemanticActionPrimitives):
    """Go to the sink object in the scene and toggle it on."""
    for i in range(ATTEMPTS):
        try:
            sink = s.scene.objects_by_category["sink"][1]
            print("Trying to NAVIGATE_TO sink.")
            execute_controller(controller._navigate_to_obj(sink), robot, s)
            print("NAVIGATE_TO sink succeeded!")
            print("Trying to TOGGLE_ON the sink.")
            execute_controller(controller.toggle_on(sink), robot, s)
            print("TOGGLE_ON the sink succeeded!")
        except ActionPrimitiveError:
            print("Attempt {} to navigate and toggle on the sink failed. Retry until {}.".format(i + 1, ATTEMPTS))
            continue

        return


def grasp_tray(s, robot, controller: StarterSemanticActionPrimitives):
    """Grasp the tray that's on the floor of the room."""
    for i in range(ATTEMPTS):
        try:
            print("Trying to GRASP tray.")
            tray = s.scene.objects_by_category["tray"][0]
            execute_controller(controller.grasp(tray), robot, s)
            print("GRASP the tray succeeded!")
        except ActionPrimitiveError:
            print("Attempt {} to grasp the tray failed. Retry until {}.".format(i + 1, ATTEMPTS))
            continue

        return


def put_on_table(s, robot, controller: StarterSemanticActionPrimitives):
    """Place the currently-held object on top of the coffee table."""
    for i in range(ATTEMPTS):
        try:
            print("Trying to PLACE_ON_TOP the held object on coffee table.")
            table = s.scene.objects_by_category["coffee_table"][0]
            execute_controller(controller.place_on_top(table), robot, s)
            print("PLACE_ON_TOP succeeded!")
        except ActionPrimitiveError:
            print("Attempt {} to place the held object failed. Retry until {}.".format(i + 1, ATTEMPTS))
            continue

        return


def open_and_close_fridge(s, robot, controller: StarterSemanticActionPrimitives):
    """Demonstrate opening and closing the fridge."""
    for i in range(ATTEMPTS):
        try:
            fridge = s.scene.objects_by_category["fridge"][0]
            print("Trying to OPEN the fridge.")
            execute_controller(controller.open(fridge), robot, s)
            print("OPEN the fridge succeeded!")
            print("Trying to CLOSE the fridge.")
            execute_controller(controller.close(fridge), robot, s)
            print("CLOSE the fridge succeeded!")
        except ActionPrimitiveError:
            print("Attempt {} to open and close the fridge failed. Retry until {}.".format(i + 1, ATTEMPTS))
            continue

        return


def open_and_close_door(s, robot, controller: StarterSemanticActionPrimitives):
    """Demonstrate opening and closing the bathroom door."""
    for i in range(ATTEMPTS):
        try:
            door = (set(s.scene.objects_by_category["door"]) & set(s.scene.objects_by_room["bathroom_0"])).pop()
            print("Trying to OPEN the door.")
            execute_controller(controller.open(door), robot, s)
            print("Trying to CLOSE the door.")
            execute_controller(controller.close(door), robot, s)
            print("CLOSE the door succeeded!")
        except ActionPrimitiveError:
            print("Attempt {} to open and close the door failed. Retry until {}.".format(i + 1, ATTEMPTS))
            continue

        return


def open_and_close_cabinet(s, robot, controller: StarterSemanticActionPrimitives):
    """Demonstrate opening and closing a drawer unit."""
    for i in range(ATTEMPTS):
        try:
            cabinet = s.scene.objects_by_category["bottom_cabinet"][2]
            print("Trying to OPEN the cabinet.")
            execute_controller(controller.open(cabinet), robot, s)
            print("Trying to CLOSE the cabinet.")
            execute_controller(controller.close(cabinet), robot, s)
            print("CLOSE the cabinet succeeded!")
        except ActionPrimitiveError:
            print("Attempt {} to open and close the cabinet failed. Retry until {}.".format(i + 1, ATTEMPTS))
            continue

        return


def main(selection="user", headless=False, short_exec=False):
    """
    Launches a simulator scene and showcases a variety of semantic action primitives such as navigation, grasping,
    placing, opening and closing.
    """
    print("*" * 80 + "\nDescription:" + main.__doc__ + "*" * 80)

    # Load the robot and place it in the scene.
    config = parse_config(os.path.join(igibson.configs_path, "behavior_robot_motion_planning.yaml"))

    config["load_object_categories"] = ["walls", "floors", "bottom_cabinet", "door", "sink", "coffee_table", "fridge"]

    env = iGibsonEnv(config_file=config, mode="gui_interactive" if not headless else "headless")

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
    env.simulator.import_object(tray)
    tray.set_position_orientation([0, 1, 0.3], p.getQuaternionFromEuler([0, np.pi / 2, 0]))

    robot = env.robots[0]

    # Create an Action Primitive Set and use it to convert high-level actions to low-level actions and execute.
    controller = StarterSemanticActionPrimitives(env=env, task=None, scene=env.scene, robot=robot)
    try:
        # The pick-and-place demo is always run.
        grasp_tray(env.simulator, robot, controller)
        put_on_table(env.simulator, robot, controller)

        # The other demos are only run in the long execution mode.
        if not short_exec:
            go_to_sink_and_toggle(env.simulator, robot, controller)
            open_and_close_fridge(env.simulator, robot, controller)
            open_and_close_door(env.simulator, robot, controller)
            open_and_close_cabinet(env.simulator, robot, controller)

        # If we're not running in headless mode, let the simulator run idle after we are done to allow user to inspect.
        if not headless:
            while True:
                action = np.zeros(robot.action_dim)
                robot.apply_action(action)
                env.simulator.step()
    finally:
        env.simulator.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
