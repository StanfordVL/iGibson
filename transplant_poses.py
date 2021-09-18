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

def transplant_urdf(old_urdf, new_urdf):
    old_tree = ET.parse(old_urdf)
    old_tree_root = old_tree.getroot()

    new_tree = ET.parse(new_urdf)
    new_tree_root = new_tree.getroot()

    object_update_dict = {}

    for element in new_tree_root:
        name = element.attrib['name']
        if name in ['fetch_gripper_robot_1', 'j_fetch_gripper_robot_1']:
            rpy = None
            xyz = None

            if element.tag == "link":
                rpy = element.attrib['rpy']
                xyz = element.attrib['xyz']
            elif element.tag == "joint":
                rpy = element.find('origin').attrib['rpy']
                xyz = element.find('origin').attrib['xyz']

            assert rpy
            assert xyz

            rpy = list(map(float, rpy.split(' ')))
            xyz = list(map(float, xyz.split(' ')))
            object_update_dict[name] = { 
              'rpy': rpy,
              'xyz': xyz,
            }

    link = ET.SubElement(old_tree_root, "link")
    link.attrib = {
        "category": "agent",
        "name": "fetch_gripper_robot_1",
        "object_scope": "agent.n.01_1",
        "rpy": " ".join([str(item) for item in rpy]),
        "xyz": " ".join([str(item) for item in xyz]),
    }
    joint = ET.SubElement(old_tree_root, "joint")
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
#     for element in old_tree_root:
#         name = element.attrib['name'] 
#         if name in object_update_dict:
#             rpy = object_update_dict[name]['rpy']
#             xyz = object_update_dict[name]['xyz']
#             if element.tag == "link":
#                 rpy_s = " ".join([str(item) for item in rpy])
#                 xyz_s = " ".join([str(item) for item in xyz])
#                 element.attrib['rpy'] = rpy_s
#                 element.attrib['xyz'] = xyz_s
#             elif element.tag == "joint":
#                 rpy_s = " ".join([str(item) for item in rpy])
#                 xyz_s = " ".join([str(item) for item in xyz])
#                 element.find('origin').attrib['rpy'] = rpy_s
#                 element.find('origin').attrib['xyz'] = xyz_s

    xmlstr = minidom.parseString(ET.tostring(old_tree_root).replace(b"\n", b"").replace(b"\t", b"")).toprettyxml()
    with open(old_urdf, "w") as f:
        f.write(xmlstr)

def main():
    args = parse_args()
    urdf_manfiest = pd.read_csv(args.manifest)
    urdfs_to_ignore = pd.read_csv("./manifest_requires_resample.csv")

    bddl.set_backend("iGibson")

    for _, row in urdf_manfiest.iterrows():
        urdf_path = row['urdf_filepath']
        if row['urdf_path'] in list(urdfs_to_ignore['demos']):
            continue
        old_urdf = os.path.join(igibson.ig_dataset_path, urdf_path)
        new_urdf = os.path.join(igibson.root_path, "data", "ig_dataset_replacement", urdf_path)
        transplant_urdf(old_urdf, new_urdf)

    # df.to_csv("qc_{}.csv".format(args.manifest[-5]))


if __name__ == "__main__":
    main()
