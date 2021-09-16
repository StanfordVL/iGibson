import argparse
import json
import logging
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import bddl
import numpy as np
import pandas as pd
import pybullet as p
from PIL import Image

from igibson.activity.activity_base import iGBEHAVIORActivityInstance
from igibson.object_states import ContactBodies
from igibson.object_states.utils import detect_collision
from igibson.robots.fetch_gripper_robot import FetchGripper
from igibson.simulator import Simulator
from igibson.utils.utils import parse_config
from igibson.external.pybullet_tools.utils import euler_from_quat, quat_from_euler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", help="manifest of files to resample")
    return parser.parse_args()


def save_modified_urdf(scene, urdf_name, robot, additional_attribs_by_name={}):
    """
    Saves a modified URDF file in the scene urdf directory having all objects added to the scene.

    :param urdf_name: Name of urdf file to save (without .urdf)
    """
    scene_tree = ET.parse(scene.scene_file)
    tree_root = scene_tree.getroot()

    xyz = robot.get_position()
    rpy = euler_from_quat(robot.get_orientation())

    for child in tree_root:
        if child.attrib['name'] == 'fetch_gripper_robot_1':
            rpy_s = " ".join([str(item) for item in rpy])
            xyz_s = " ".join([str(item) for item in xyz])
            child.attrib['rpy'] = rpy_s
            child.attrib['xyz'] = xyz_s
        elif child.attrib['name'] == 'j_fetch_gripper_robot_1':
            rpy_s = " ".join([str(item) for item in rpy])
            xyz_s = " ".join([str(item) for item in xyz])
            child.find('origin').attrib['rpy'] = rpy_s
            child.find('origin').attrib['xyz'] = xyz_s

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
    needs_adjusted_orientation = []
    needs_resample = []

    for _, row in urdf_manfiest.iterrows():
        # task = row["task"]
        # task_id = row["task_ids"]
        # scene_id = row["scene_id"]
        # init_id = row["init_ids"]
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
        adjusted_orientation = False

        success = igbhvr_act_inst.initialize_simulator(
            simulator=simulator,
            scene_id=scene_id,
            mode="iggui",
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

        if success:
            robot = simulator.robots[0]
            # p.resetDebugVisualizerCamera(cameraDistance=1.0, cameraYaw=0.0, cameraPitch=0.0, cameraTargetPosition=robot.get_position())
            x, y, z = robot.get_position()
            simulator.viewer.px = x - 0.5
            simulator.viewer.py = y + 0.2
            simulator.viewer.pz = z + 0.2
            simulator.sync()
            import pdb; pdb.set_trace()
            for i in range(250):
                simulator.viewer.update()
            original_robot_position = robot.get_position()
            robot.set_position(
                [original_robot_position[0], original_robot_position[1], original_robot_position[2] + 0.02]
            )
            robot.robot_specific_reset()
            robot_collision = detect_collision(robot.get_body_id())
            valid = not robot_collision
            if valid:
                # Fall for 0.1 second to settle down
                for _ in range(3):
                    robot.apply_action(np.zeros(11))
                    simulator.step()

                # Ugly hack to manually update the robot state because it's not updated by simulator.step()
                robot.states[ContactBodies].update()
                valid = onfloor_condition.evaluate()
            # print("check valid")
            # embed()
            if not valid:
                # import pdb; pdb.set_trace()
                # Try to adjust the position of the robot
                for i in range(2000):
                    new_position = original_robot_position + rng.normal(0, scale=0.1, size=3)
                    new_position[2] = original_robot_position[2] + 0.02
                    robot.set_position(new_position)
                    if i > 1000:
                        adjusted_orientation = True
                        new_orientation = quat_from_euler([0, 0, rng.uniform(-np.pi, np.pi)])
                        robot.set_orientation(new_orientation)
                    robot.robot_specific_reset()
                    robot_collision = detect_collision(robot.get_body_id())
                    if robot_collision:
                        continue

                    # Fall for 0.1 second to settle down
                    for _ in range(3):
                        robot.apply_action(np.zeros(11))
                        simulator.step()

                    # Ugly hack to manually update the robot state because it's not updated by simulator.step()
                    robot.states[ContactBodies].update()
                    adjusted = onfloor_condition.evaluate()
                    if adjusted:
                        break
            # print("check adjusted")
            # embed()
            if (not valid) and (not adjusted):
                adjusted_orientation = True
                for i in range(50):
                    resampled = onfloor_condition.children[0].sample(True)
                success = resampled
            # print("check resampled")
            # embed()

        # snapshot("cache_images/{}.png".format(urdf_path), simulator)

        original_valid.append(valid)
        needs_adjustment.append(adjusted)
        needs_adjusted_orientation.append(adjusted_orientation)
        needs_resample.append(resampled)
        scene_successful.append(success)
        urdf_list.append(urdf_path)

        # if success:
        #     save_modified_urdf(simulator.scene, urdf_path, robot)

        simulator.disconnect()

        df = pd.DataFrame(
            {
                "urdf": urdf_list,
                "success": scene_successful,
                "valid": original_valid,
                "adjusted_position": needs_adjustment,
                "adjusted_orientation": needs_adjusted_orientation,
                "resampled": needs_resample,
            }
        )
        # df.to_csv("qc_{}.csv".format(args.manifest[-5]))


if __name__ == "__main__":
    main()
