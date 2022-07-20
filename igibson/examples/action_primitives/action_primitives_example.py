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
    try:
        sink = s.scene.objects_by_category["sink"][1]
        print("Trying to NAVIGATE_TO sink.")
        execute_controller(controller._navigate_to_obj(sink), robot, s)
        print("NAVIGATE_TO sink succeeded!")
        print("Trying to TOGGLE_ON the sink.")
        execute_controller(controller.toggle_on(sink), robot, s)
        print("TOGGLE_ON the sink succeeded!")
        return True
    except ActionPrimitiveError:
        print("TOGGLE_ON the sink failed!")
        return False


def grasp_tray(s, robot, controller: StarterSemanticActionPrimitives):
    """Grasp the tray that's on the floor of the room."""
    try:
        print("Trying to GRASP tray.")
        tray = s.scene.objects_by_category["tray"][0]
        execute_controller(controller.grasp(tray), robot, s)
        print("GRASP the tray succeeded!")
        return True
    except ActionPrimitiveError as ape:
        print("GRASP the tray failed!")
        return False


def put_on_table(s, robot, controller: StarterSemanticActionPrimitives):
    """Place the currently-held object on top of the coffee table."""
    try:
        print("Trying to PLACE_ON_TOP the held object on coffee table.")
        table = s.scene.objects_by_category["coffee_table"][0]
        execute_controller(controller.place_on_top(table), robot, s)
        print("PLACE_ON_TOP succeeded!")
        return True
    except ActionPrimitiveError:
        print("PLACE_ON_TOP failed!")
        return False


def open_and_close_fridge(s, robot, controller: StarterSemanticActionPrimitives):
    """Demonstrate opening and closing the fridge."""
    try:
        fridge = s.scene.objects_by_category["fridge"][0]
        print("Trying to OPEN the fridge.")
        execute_controller(controller.open(fridge), robot, s)
        print("OPEN the fridge succeeded!")
        print("Trying to CLOSE the fridge.")
        execute_controller(controller.close(fridge), robot, s)
        print("CLOSE the fridge succeeded!")
        return True
    except ActionPrimitiveError:
        print("OPEN and CLOSE fridge failed!")
        return False


def open_and_close_door(s, robot, controller: StarterSemanticActionPrimitives):
    """Demonstrate opening and closing the bathroom door."""
    try:
        door = (set(s.scene.objects_by_category["door"]) & set(s.scene.objects_by_room["bathroom_0"])).pop()
        print("Trying to OPEN the door.")
        execute_controller(controller.open(door), robot, s)
        print("Trying to CLOSE the door.")
        execute_controller(controller.close(door), robot, s)
        print("CLOSE the door succeeded!")
        return True
    except ActionPrimitiveError:
        print("OPEN and CLOSE door failed!")
        return False


def open_and_close_cabinet(s, robot, controller: StarterSemanticActionPrimitives):
    """Demonstrate opening and closing a drawer unit."""
    try:
        cabinet = s.scene.objects_by_category["bottom_cabinet"][2]
        print("Trying to OPEN the cabinet.")
        execute_controller(controller.open(cabinet), robot, s)
        print("Trying to CLOSE the cabinet.")
        execute_controller(controller.close(cabinet), robot, s)
        print("CLOSE the cabinet succeeded!")
        return True
    except ActionPrimitiveError:
        print("OPEN and CLOSE cabinet failed!")
        return False


def open_hand(s, controller, robot):
    action = np.zeros(robot.action_dim)
    action[robot.controller_action_idx["gripper_right_hand"]] = 1.0
    for _ in range(10):
        robot.apply_action(action)
        s.step()

    for _ in range(10):
        execute_controller(controller.reset(), robot, s)


def reset(env, robot, controller, tray):
    # TODO: there is still something wrong here. After reset, is like the hands collide with the tray, even if it is not
    #  supposed to be there
    print("Resetting scene")
    tray.set_position_orientation([0, 1, 0.3], p.getQuaternionFromEuler([0, np.pi / 2, 0]))
    env.land(robot, [0, 0, 0], [0, 0, 0])
    robot.reset()
    tray.set_position_orientation([0, 1, 0.3], p.getQuaternionFromEuler([0, np.pi / 2, 0]))
    action = np.zeros(robot.action_dim)
    robot.apply_action(action)
    env.simulator.step()
    open_hand(env.simulator, controller, robot)
    robot.set_position_orientation(*robot.get_parts["body"].get_position_orientation())
    cam_pose = robot.get_parts["eye"].get_position_orientation()
    # Rotate cam
    cam_pose_new = (cam_pose[0], p.getQuaternionFromEuler([0, 70, 0]))
    robot.get_parts["eye"].set_position_orientation(*cam_pose_new)
    # Make the viewer follow the robot, placing the virtual camera in front of it and watching it
    if env.simulator.viewer is not None:
        env.simulator.viewer.following_viewer = True
        env.simulator.viewer.camlocation_in_rf = np.array([1.0, 0.0, 1])  # x is in front of the robot
    tray.set_position_orientation([0, 1, 0.3], p.getQuaternionFromEuler([0, np.pi / 2, 0]))
    env.simulator.step()


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

    robot = env.robots[0]
    # Create an Action Primitive Set and use it to convert high-level actions to low-level actions and execute.
    controller = StarterSemanticActionPrimitives(env=env, task=None, scene=env.scene, robot=robot)

    inspect_effect = False  # Change to true if you are manually executing and want to see the effect of actions

    for attempt in range(20):
        reset(env, robot, controller, tray)
        print("Attempt {}".format(attempt))
        if grasp_tray(env.simulator, robot, controller):
            print("Grasped object. Trying to put it on the table.")
            if put_on_table(env.simulator, robot, controller):
                print("Placed object.")
                # If we're not running in headless mode, let the simulator run idle after we are done to allow user to inspect.
                if not headless and inspect_effect:
                    while 1000:
                        action = np.zeros(robot.action_dim)
                        robot.apply_action(action)
                        env.simulator.step()
                break

    # The other demos are only run in the long execution mode.
    if not short_exec:
        functions_to_execute = [
            go_to_sink_and_toggle,
            open_and_close_fridge,
            open_and_close_door,
            open_and_close_cabinet,
        ]
        print("Attempting a second sequence of actions")

        for function in functions_to_execute:
            print("Executing another function")
            for attempt in range(5):
                reset(env, robot, controller, tray)
                print("Attempt {}".format(attempt))
                input("[debug] press enter")
                if function(env.simulator, robot, controller):
                    print("Success!")
                    # If we're not running in headless mode, let the simulator run idle after we are done to allow user to inspect.
                    if not headless and inspect_effect:
                        while 1000:
                            action = np.zeros(robot.action_dim)
                            robot.apply_action(action)
                            env.simulator.step()
                else:
                    print("Failure")

    env.simulator.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.getLogger(name="igibson.action_primitives.starter_semantic_action_primitives").setLevel(level=logging.DEBUG)
    logging.getLogger(name="igibson.utils.motion_planning_utils").setLevel(level=logging.DEBUG)
    main()
