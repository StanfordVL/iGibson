import argparse
import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import bddl
import pandas as pd

import igibson
from igibson.external.pybullet_tools.utils import euler_from_quat

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", help="manifest of files to resample")
    return parser.parse_args()


def save_modified_urdf(urdf_path, additional_attribs_by_name={}):
    """
    Saves a modified URDF file in the scene urdf directory having all objects added to the scene.

    :param urdf_name: Name of urdf file to save (without .urdf)
    """
    scene_tree = ET.parse(os.path.join(igibson.ig_dataset_path, urdf_path))
    tree_root = scene_tree.getroot()

    for child in tree_root:
        if child.attrib['name'] in ['fetch_gripper_robot_1', 'j_fetch_gripper_robot_1']:
            tree_root.remove(child)
            import pdb; pdb.set_trace()

    xmlstr = minidom.parseString(ET.tostring(tree_root).replace(b"\n", b"").replace(b"\t", b"")).toprettyxml()
    with open(os.path.join(igibson.ig_dataset_path, urdf_path), "w") as f:
        f.write(xmlstr)

def main():
    args = parse_args()
    urdf_manfiest = pd.read_csv(args.manifest)

    bddl.set_backend("iGibson")

    for _, row in urdf_manfiest.iterrows():
        info = row['demos'].split('_')
        scene_id = '_'.join(info[0:3])
        task = '_'.join(info[4:-2])
        init_id = info[-1]
        task_id = 0
        urdf_path = os.path.join(igibson.ig_dataset_path, 'scenes', scene_id, 'urdf', '{}_task_{}_{}_{}.urdf'.format(scene_id, task, task_id, init_id))
        save_modified_urdf(urdf_path)

    # df.to_csv("qc_{}.csv".format(args.manifest[-5]))


if __name__ == "__main__":
    main()
