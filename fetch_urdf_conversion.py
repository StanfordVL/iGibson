import argparse
import json
import logging
import os

import bddl
import pybullet as p
from PIL import Image
import numpy as np
import xml.etree.ElementTree as ET

import pandas as pd
from igibson.utils.utils import parse_config
from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.simulator import Simulator
from igibson.robots.fetch_gripper_robot import FetchGripper

from xml.dom import minidom


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_id", type=int, default=0, help="BDDL integer ID, matching suffix of bddl.")
    parser.add_argument("--max_trials", type=int, default=1, help="Maximum number of trials to try sampling.")
    parser.add_argument(
        "--num_initializations", type=int, default=10, help="Number of initialization per BDDL per scene."
    )
    parser.add_argument("--start_initialization", type=int, default=0, help="Starting idx for initialization")
    return parser.parse_args()


def detect_collisions(robot):
    for contact in p.getContactPoints(robot.get_body_id()):
        if contact[8] <= -0.0007 and contact[2] != 3:
            return True
    return False


def save_modified_urdf(scene, urdf_name, robot, additional_attribs_by_name={}):
    """
    Saves a modified URDF file in the scene urdf directory having all objects added to the scene.

    :param urdf_name: Name of urdf file to save (without .urdf)
    """
    scene_tree = ET.parse(scene.scene_file)
    tree_root = scene_tree.getroot()

    robot.name = "fetch_gripper_robot_1"
    robot.body_id = robot.get_body_id()
    robot.get_position_orientation = lambda: (robot.get_position(), robot.get_orientation())
    scene.save_obj_or_multiplexer(robot, tree_root, additional_attribs_by_name)
    path_to_urdf = os.path.join(scene.scene_dir, "urdf", urdf_name + ".urdf")
    xmlstr = minidom.parseString(ET.tostring(tree_root).replace(b"\n", b"").replace(b"\t", b"")).toprettyxml()
    with open(path_to_urdf, "w") as f:
        f.write(xmlstr)


def snapshot(filename, simulator):
    # Take a picture of the old position
    camera_pos = simulator.robots[0].get_position()
    offset = np.array([1, 0, 1.5])
    camera_pos += offset
    viewing_direction = np.array([-1, 0, -0.75])

    # simulator.viewer.px = camera_pos[0]
    # simulator.viewer.py = camera_pos[1]
    # simulator.viewer.pz = camera_pos[2]
    # simulator.viewer.view_direction = viewing_direction

    simulator.sync()
    simulator.renderer.set_camera(camera_pos, camera_pos + viewing_direction, [0, 0, 1])
    rgb = simulator.renderer.render(modes=("rgb"))[0][:, :, :3]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(filename)


def main():
    args = parse_args()
    task_id = args.task_id
    start_initialization = args.start_initialization
    num_initializations = args.num_initializations

    bddl.set_backend("iGibson")

    scene_json = os.path.join(os.path.dirname(bddl.__file__), "../utils", "activity_to_preselected_scenes.json")

    with open(scene_json) as f:
        activity_to_scenes = json.load(f)

    urdf_list = []
    original_valid = []
    scene_successful = []
    needs_adjustment = []
    needs_resample = []
    for task in activity_to_scenes:
        scene_choices = activity_to_scenes[task]
        if task not in os.listdir("/home/michael/Repositories/lab/bddl/bddl/activity_conditions"):
            continue

        logging.warning("TASK: {}".format(task))
        logging.warning("TASK ID: {}".format(task_id))
        logging.warning(("SCENE CHOICES", scene_choices))

        simulator = Simulator(mode="headless", image_width=960, image_height=720, device_idx=0)

        scene_kwargs = {}

        rng = np.random.default_rng()
        config = parse_config("igibson/examples/configs/behavior_onboard_sensing_fetch.yaml")
        igbhvr_act_inst = iGBEHAVIORActivityInstance(
            task, activity_definition=task_id, robot_type=FetchGripper, robot_config=config
        )
        for scene_id in scene_choices:
            logging.warning(("TRY SCENE:", scene_id))

            for init_id in range(start_initialization, start_initialization + num_initializations):
                urdf_path = "{}_task_{}_{}_{}".format(scene_id, task, task_id, init_id)
                success = False
                valid = False
                adjusted = False
                resampled = False

                scene_kwargs = {
                    "urdf_file": urdf_path,
                }
                success = igbhvr_act_inst.initialize_simulator(
                    simulator=simulator,
                    scene_id=scene_id,
                    mode="headless",
                    load_clutter=True,
                    should_debug_sampling=False,
                    scene_kwargs=scene_kwargs,
                    online_sampling=False,
                )
                robot = simulator.robots[0]
                robot.apply_action(np.zeros(11))
                try:
                    simulator.step()
                except:
                    import pdb

                    pdb.set_trace()
                robot.robot_specific_reset()

                robot_collision = detect_collisions(robot)
                found_good_position = False

                if robot_collision:
                    snapshot("cache_images/pre_{}.png".format(urdf_path), simulator)
                    preAdjustState = p.saveState()
                    original_robot_position = robot.get_position()

                    found_good_position = False

                    # Try to adjust the position of the robot
                    for _ in range(1000):
                        new_position = original_robot_position + rng.normal(0, scale=0.5, size=3)
                        new_position[2] = original_robot_position[2] + 0.02
                        robot.set_position(new_position)

                        found_good_position = True
                        for i in range(25):
                            if i == 0:
                                p.resetBaseVelocity(
                                    robot.get_body_id(), linearVelocity=[0, 0, 0], angularVelocity=[0, 0, 0]
                                )
                            robot.robot_specific_reset()
                            try:
                                simulator.step()
                            except:
                                import pdb

                                pdb.set_trace()
                            if detect_collisions(robot):
                                found_good_position = False
                                break

                        for condition in igbhvr_act_inst.initial_conditions:
                            if simulator.robots[0] in condition.get_relevant_objects():
                                found_good_position = found_good_position and condition.evaluate()

                        if not found_good_position:
                            p.restoreState(preAdjustState)
                            continue
                        else:
                            adjusted = True
                            break

                    if not found_good_position:
                        for condition in igbhvr_act_inst.initial_conditions:
                            if simulator.robots[0] in condition.get_relevant_objects():
                                condition.children[0].sample(True)
                                resampled = True

                    snapshot("cache_images/post_{}.png".format(urdf_path), simulator)

                original_valid.append(valid)
                needs_adjustment.append(adjusted)
                needs_resample.append(resampled)
                scene_successful.append(success)
                urdf_list.append(urdf_path)

                df = pd.DataFrame(
                    {
                        "urdf": urdf_list,
                        "success": scene_successful,
                        "resampled": needs_resample,
                        "adjusted": needs_adjustment,
                    }
                )
                df.to_csv("qc.csv")

                save_modified_urdf(simulator.scene, urdf_path, robot)


if __name__ == "__main__":
    main()
