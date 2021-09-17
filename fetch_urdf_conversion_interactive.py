import argparse
import json
import logging
import os
import pdb
import xml.etree.ElementTree as ET
from xml.dom import minidom

import bddl
import numpy as np
import pandas as pd
import pybullet as p
from IPython import embed
from PIL import Image

from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.object_states import ContactBodies
from igibson.object_states.utils import detect_collision
from igibson.robots.fetch_gripper_robot import FetchGripper
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config
from igibson.external.pybullet_tools.utils import euler_from_quat
from igibson.utils.constants import NON_SAMPLEABLE_OBJECTS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", help="manifest of files to resample")
    return parser.parse_args()


def save_modified_urdf(urdf_name, robot, scene, activity, additional_attribs_by_name={}):
    """
    Saves a modified URDF file in the scene urdf directory having all objects added to the scene.

    :param urdf_name: Name of urdf file to save (without .urdf)
    """
    scene_tree = ET.parse(scene.scene_file)
    tree_root = scene_tree.getroot()
    object_update_dict = {}
    for obj_name, obj in activity.object_scope.items():
        if obj_name.split('_')[0] not in NON_SAMPLEABLE_OBJECTS:
            if 'agent' not in obj.category:
                object_update_dict[obj.name] = {
                    "rpy": euler_from_quat(obj.get_orientation()),
                    "xyz": obj.get_position(),
                }
                object_update_dict['j_' + obj.name] = {
                    "rpy": euler_from_quat(obj.get_orientation()),
                    "xyz": obj.get_position(),
                }
    for element in tree_root:
        name = element.attrib['name'] 
        if name in object_update_dict:
            rpy = object_update_dict[name]['rpy']
            xyz = object_update_dict[name]['xyz']
            if element.tag == "link":
                rpy_s = " ".join([str(item) for item in rpy])
                xyz_s = " ".join([str(item) for item in xyz])
                element.attrib['rpy'] = rpy_s
                element.attrib['xyz'] = xyz_s
            elif element.tag == "joint":
                rpy_s = " ".join([str(item) for item in rpy])
                xyz_s = " ".join([str(item) for item in xyz])
                element.find('origin').attrib['rpy'] = rpy_s
                element.find('origin').attrib['xyz'] = xyz_s

    xyz = robot.get_position()
    rpy = euler_from_quat(robot.get_orientation())
    link = ET.SubElement(tree_root, "link")
    link.attrib = {
        "category": "agent",
        "name": "fetch_gripper_robot_1",
        "object_scope": "agent.n.01_1",
        "rpy": " ".join([str(item) for item in rpy]),
        "xyz": " ".join([str(item) for item in xyz]),
    }
    joint = ET.SubElement(tree_root, "joint")
    joint.attrib = {
        "name": "j_fetch_gripper_robot_1",
        "type": "floating",
    }
    origin = ET.SubElement(joint, "origin")
    origin.attrib = {
        "rpy": " ".join([str(item) for item in rpy]),
        "xyz": " ".join([str(item) for item in xyz]),
    }
    child = ET.SubElement(joint, "child")
    child.attrib = {
        "link": "fetch_gripper_robot_1",
    }
    parent = ET.SubElement(joint, "parent")
    parent.attrib = {
        "link": "world",
    }
    path_to_urdf = os.path.join(scene.scene_dir, "urdf", urdf_name + ".urdf")
    xmlstr = minidom.parseString(ET.tostring(tree_root).replace(b"\n", b"").replace(b"\t", b"")).toprettyxml()
    with open(path_to_urdf, "w") as f:
        f.write(xmlstr)
    print(path_to_urdf)


def snapshot(filename, simulator):
    camera_pos = simulator.robots[0].get_position()
    offset = np.array([1, 0, 1.5])
    camera_pos += offset
    viewing_direction = np.array([-1, 0, -0.75])
    simulator.sync()
    simulator.renderer.set_camera(camera_pos, camera_pos + viewing_direction, [0, 0, 1])
    rgb = simulator.renderer.render(modes=("rgb"))[0][:, :, :3]
    Image.fromarray((rgb * 255).astype(np.uint8)).save(filename)


def main():
    args = parse_args()
    urdf_manfiest = pd.read_csv(args.manifest)

    rng = np.random.default_rng()

    bddl.set_backend("iGibson")

    urdf_list = []
    original_valid = []
    scene_successful = []
    needs_adjustment = []
    needs_resample = []

    for _, row in urdf_manfiest.iterrows():
        info = row['demos'].split('_')
        scene_id = '_'.join(info[0:3])
        task = '_'.join(info[4:-2])
        init_id = info[-1]
        task_id = 0

        logging.warning("TASK: {}".format(task))
        logging.warning("TASK ID: {}".format(task_id))
        simulator = Simulator(mode="iggui", image_width=960, image_height=720, device_idx=0)

        config = parse_config("igibson/examples/configs/behavior_onboard_sensing_fetch.yaml")
        igbhvr_act_inst = iGBEHAVIORActivityInstance(
            task, activity_definition=task_id, robot_type=FetchGripper, robot_config=config
        )

        urdf_path = "{}_task_{}_{}_{}".format(scene_id, task, task_id, init_id)
        success = False
        valid = False
        adjusted = False
        resampled = False

        success = igbhvr_act_inst.initialize_simulator(
            simulator=simulator,
            scene_id=scene_id,
            mode="headless",
            load_clutter=True,
            should_debug_sampling=False,
            scene_kwargs={"urdf_file": urdf_path},
            online_sampling=False,
        )

        onfloor_condition = None
        for condition in igbhvr_act_inst.initial_conditions:
            if simulator.robots[0] in condition.get_relevant_objects():
                onfloor_condition = condition
                break
        assert onfloor_condition is not None

        def settle():
            for _ in range(200):
                robot.apply_action(np.zeros(11))
                simulator.step()

            # Ugly hack to manually update the robot state because it's not updated by simulator.step()
            robot.states[ContactBodies].update()
            valid = onfloor_condition.evaluate()
            return valid

        robot = simulator.robots[0]
        # original_robot_position = robot.get_position()
        # robot.set_position([100, 100, 100])
        # robot.robot_specific_reset()
        # robot_collision = detect_collision(robot.get_body_id())
        # settle()

        # robot_collision = detect_collision(robot.get_body_id())
        # def check_onfloor():
        #     robot.states[ContactBodies].update()
        #     adjusted = onfloor_condition.evaluate()
        #     return adjusted
        # onfloor = check_onfloor()

        # snapshot("cache_images/{}.png".format(urdf_path), simulator)

        original_valid.append(valid)
        needs_adjustment.append(adjusted)
        needs_resample.append(resampled)
        scene_successful.append(success)
        urdf_list.append(urdf_path)

        # good_pos = [3.61623098e+00, 9.83628929e+00, 7.23093368e-03]
        # for _ in range(1000):
        #     simulator.step()
        # robot.set_position(good_pos)
        # robot.robot_specific_reset()
        # robot.apply_action(np.zeros(11))
        # settle()
        # simulator.sync()
        # if success:
        # simulator.scene.save_modified_urdf(urdf_path)

        for condition in igbhvr_act_inst.initial_conditions:
            if len(condition.body) == 3:
                obj = condition.scope[condition.children[0].input1]
                if obj.category != "agent":

                    obj.force_wakeup()
                    obj.set_position([300, np.random.randint(-10, 10), 300])

        simulator.sync()
        for condition in igbhvr_act_inst.initial_conditions:
            if len(condition.body) == 3:
                obj = condition.scope[condition.children[0].input1]
                if obj.category == "agent":
                    condition.children[0].sample(True)
                    robot.robot_specific_reset()
                    robot.apply_action(np.zeros(11))

        for condition in igbhvr_act_inst.initial_conditions:
            if len(condition.body) == 3:
                obj = condition.scope[condition.children[0].input1]
                if obj.category != "agent":
                    condition.children[0].sample(True)

        camera_pos = simulator.robots[0].get_position()
        offset = np.array([1, 0, 1.5])
        camera_pos += offset
        viewing_direction = np.array([-1, 0, -0.75])

        simulator.viewer.px = camera_pos[0]
        simulator.viewer.py = camera_pos[1]
        simulator.viewer.pz = camera_pos[2]
        simulator.viewer.view_direction = viewing_direction

        settle()

        save_modified_urdf(urdf_path, robot, simulator.scene, igbhvr_act_inst)
        snapshot("cache_images/{}.png".format(urdf_path), simulator)

        simulator.disconnect()

        df = pd.DataFrame(
            {
                "urdf": urdf_list,
                "success": scene_successful,
                "valid": original_valid,
                "adjusted": needs_adjustment,
                "resampled": needs_resample,
            }
        )
        df.to_csv("qc_{}.csv".format(args.manifest[-5]))


if __name__ == "__main__":
    main()
