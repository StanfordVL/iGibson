import os

import numpy as np
import pybullet as p

from igibson.action_generators.motion_primitive_generator import MotionPrimitiveActionGenerator
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


def go_to_sink_and_toggle(s, robot, controller: MotionPrimitiveActionGenerator):
    sink = s.scene.objects_by_category["sink"][1]
    execute_controller(controller._navigate_to_obj(sink), robot, s)
    execute_controller(controller.toggle_on(sink), robot, s)


def grasp_tray(s, robot, controller: MotionPrimitiveActionGenerator):
    tray = s.scene.objects_by_category["tray"][0]
    execute_controller(controller.grasp(tray), robot, s)


def put_on_table(s, robot, controller: MotionPrimitiveActionGenerator):
    table = s.scene.objects_by_category["coffee_table"][0]
    execute_controller(controller.place_on_top(table), robot, s)


def open_and_close_fridge(s, robot, controller: MotionPrimitiveActionGenerator):
    fridge = s.scene.objects_by_category["fridge"][0]
    execute_controller(controller.open(fridge), robot, s)
    execute_controller(controller.close(fridge), robot, s)


def open_and_close_door(s, robot, controller: MotionPrimitiveActionGenerator):
    door = (set(s.scene.objects_by_category["door"]) & set(s.scene.objects_by_room["bathroom_0"])).pop()
    execute_controller(controller.open(door), robot, s)
    execute_controller(controller.close(door), robot, s)


def open_and_close_cabinet(s, robot, controller: MotionPrimitiveActionGenerator):
    cabinet = s.scene.objects_by_category["bottom_cabinet"][2]
    execute_controller(controller.open(cabinet), robot, s)
    execute_controller(controller.close(cabinet), robot, s)


def main():
    s = Simulator(mode="gui_non_interactive", image_width=512, image_height=512, device_idx=0, use_pb_gui=True)
    scene = InteractiveIndoorScene(
        "Rs_int", load_object_categories=["walls", "floors", "bottom_cabinet", "door", "sink", "coffee_table", "fridge"]
    )
    s.import_scene(scene)

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

    robot = BehaviorRobot(s)
    s.import_robot(robot)
    robot.set_position_orientation([0, 0, 1], [0, 0, 0, 1])

    for _ in range(300):
        s.step()

    controller = MotionPrimitiveActionGenerator(None, scene, robot)

    try:
        # go_to_sink_and_toggle(s, robot, controller)
        grasp_tray(s, robot, controller)
        put_on_table(s, robot, controller)
        # open_and_close_fridge(s, robot, controller)
        # open_and_close_door(s, robot, controller)
        # open_and_close_cabinet(s, robot, controller)

        while True:
            action = np.zeros(28)
            robot.apply_action(action)
            s.step()
    finally:
        s.disconnect()


if __name__ == "__main__":
    main()
