import os

import numpy as np
import pybullet as p

from igibson import object_states
from igibson.examples.mp_replay.behavior_grasp_planning_utils import get_grasp_poses_for_object
from igibson.examples.mp_replay.behavior_motion_primitive_controller import MotionPrimitiveController
from igibson.objects.articulated_object import URDFObject
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
from igibson.utils.assets_utils import get_ig_avg_category_specs, get_ig_model_path


def execute_controller(ctrl_gen, robot, s):
    for action in ctrl_gen:
        new_action = np.zeros(28)  # Add the reset dimensions
        new_action[:19] = action[:19]
        new_action[20:27] = action[19:]
        robot.apply_action(new_action)
        s.step()


def go_to_waypoint(s, robot, controller: MotionPrimitiveController):
    # Navigate 1m back and 1m to the side.
    delta_xy = np.array([-1, -1])
    robot_xy = np.array(robot.parts["body"].get_position())[:2]
    waypoint = robot_xy + delta_xy
    execute_controller(controller._navigate_to_pose_direct(waypoint), robot, s)


def go_to_sink_and_toggle(s, robot, controller: MotionPrimitiveController):
    sink = s.scene.objects_by_category["sink"][1]
    execute_controller(controller._navigate_to_obj(sink), robot, s)
    execute_controller(controller.toggle_on(sink), robot, s)


def prepare_to_grasp_tray(s, robot, controller: MotionPrimitiveController):
    tray = s.scene.objects_by_category["tray"][0]
    execute_controller(controller._navigate_to_obj(tray), robot, s)

    for grasp_pose, _ in get_grasp_poses_for_object(robot, tray):
        robot.parts["right_hand"].set_position_orientation(*grasp_pose)
        # execute_controller(controller._move_hand(grasp_pose), robot, s)


def grasp_tray(s, robot, controller: MotionPrimitiveController):
    tray = s.scene.objects_by_category["tray"][0]
    execute_controller(controller.grasp(tray), robot, s)


def put_on_table(s, robot, controller: MotionPrimitiveController):
    table = s.scene.objects_by_category["coffee_table"][0]
    execute_controller(controller.place_on_top(table), robot, s)


def hand_fwd_by_one(s, robot, controller: MotionPrimitiveController):
    execute_controller(controller._move_hand_direct_relative_to_robot(([0.5, 0, 0], [0, 0, 0, 1])), robot, s)


def main():
    s = Simulator(mode="gui", image_width=512, image_height=512, device_idx=0)
    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=["walls", "floors", "bed", "door", "sink", "coffee_table"]
    )
    s.import_ig_scene(scene)

    coffee_table = scene.objects_by_category["coffee_table"][0]

    model_path = get_ig_model_path("tray", "tray_000")
    model_filename = os.path.join(model_path, "tray_000.urdf")
    max_bbox = [0.1, 0.1, 0.1]
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

    # tray.states[object_states.OnTop].set_value(coffee_table, True, use_ray_casting_method=True)

    robot = BehaviorRobot(s)
    s.import_behavior_robot(robot)
    robot.set_position_orientation([0, 0, 1], [0, 0, 0, 1])
    robot.activate()

    for _ in range(100):
        s.step()

    controller = MotionPrimitiveController(scene, robot)

    try:
        # go_to_waypoint(s, robot, controller)
        go_to_sink_and_toggle(s, robot, controller)
        # hand_fwd_by_one(s, robot, controller)
        # prepare_to_grasp_tray(s, robot, controller)
        # grasp_tray(s, robot, controller)
        # put_on_table(s, robot, controller)

        while True:
            action = np.zeros(28)
            robot.apply_action(action)
            s.step()
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
