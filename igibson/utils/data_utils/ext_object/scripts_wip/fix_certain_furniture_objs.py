import os
import xml.etree.ElementTree as ET
from xml.dom import minidom

import igibson

fixed_categories = frozenset(
    [
        "bottom_cabinet",
        "bottom_cabinet_no_top",
        "top_cabinet",
        "shelf",
        "pool_table",
        "stand",
        "pedestal_table",
        "coffee_table",
        "breakfast_table",
        "console_table",
        "desk",
        "gaming_table",
    ]
)

scene_root_dir = os.path.join(igibson.ig_dataset_path, "scenes")
for scene_id in os.listdir(scene_root_dir):
    # if 'Rs_int' not in scene_id:
    #     continue
    if "_int" not in scene_id:
        continue
    scene_urdf_dir = os.path.join(scene_root_dir, scene_id, "urdf")
    for urdf_file in os.listdir(scene_urdf_dir):
        if "_task_" not in urdf_file:
            continue
        # if '_task_' not in urdf_file:
        #     continue
        if "fixed_furniture" in urdf_file:
            continue
        urdf_file = os.path.join(scene_urdf_dir, urdf_file)
        root = ET.parse(urdf_file)
        for joint in root.findall("joint"):
            child = joint.find("child")
            link = root.find('link[@name="{}"]'.format(child.attrib["link"]))
            if link.attrib["category"] in fixed_categories:
                joint.attrib["type"] = "fixed"

        tree_root = root.getroot()

        path_to_urdf = urdf_file.replace(".urdf", "_fixed_furniture.urdf")
        xmlstr = minidom.parseString(ET.tostring(tree_root).replace(b"\n", b"").replace(b"\t", b"")).toprettyxml()
        with open(path_to_urdf, "w+") as f:
            f.write(xmlstr)

        print(path_to_urdf)
