import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

import cv2
import numpy as np
import pybullet as p
import trimesh
from IPython import embed
from scipy.spatial.transform import Rotation as R

root_folder = "articulated_objects"
processed_folder = "/cvgl2/u/chengshu/gibsonv2/igibson/data/ig_dataset/objects"
scene_name = "test_scene"
scene_processed_folder = "/cvgl2/u/chengshu/gibsonv2/igibson/data/ig_dataset/scenes"


def build_object_hierarchy():
    parent_to_children = dict()
    for obj in os.listdir(root_folder):
        obj_folder = os.path.join(root_folder, obj)
        if not os.path.isdir(obj_folder):
            continue
        tokens = obj.split("-")

        # Base link
        if len(tokens) == 3:
            obj_cat, obj_inst_id, link_name = tokens
            parent_link_name, joint_type, joint_limit = None, None, None

        # Non-base link
        else:
            obj_cat, obj_inst_id, link_name, parent_link_name, joint_type, joint_limit = tokens

        obj_inst = (obj_cat, obj_inst_id)
        if obj_inst not in parent_to_children:
            parent_to_children[obj_inst] = {
                "hierarchy": dict(),
            }

        if parent_link_name is None:
            parent_to_children[obj_inst]["hierarchy"]["root"] = set([(link_name, None)])
        else:
            if parent_link_name not in parent_to_children[obj_inst]["hierarchy"]:
                parent_to_children[obj_inst]["hierarchy"][parent_link_name] = set()
            parent_to_children[obj_inst]["hierarchy"][parent_link_name].add((link_name, joint_type))
    return parent_to_children


def fix_mtl_file(obj_dir, mtl_file):
    mtl_file = os.path.join(obj_dir, mtl_file)
    new_lines = []
    with open(mtl_file) as f:
        for line in f.readlines():
            if "map" in line:
                line = line.replace("material\\", "material/")
                map_path = os.path.join(obj_dir, line.split(" ")[1].strip())
                # For some reason, pybullet won't load the texture files unless we save it again with OpenCV
                img = cv2.imread(map_path)
                cv2.imwrite(map_path, img)

            new_lines.append(line)

    with open(mtl_file, "w") as f:
        for line in new_lines:
            f.write(line)


def scale_rotate_mesh(mesh, translation, rotation):
    # Correct coordinate system change in pybullet
    coordinate_matrix = np.eye(4)
    coordinate_matrix[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()

    # Original mesh 1 unit = 1mm
    scale_matrix = trimesh.transformations.scale_matrix(0.001)

    mesh.apply_transform(scale_matrix).apply_transform(coordinate_matrix)

    # When translation is None, assume the current mesh is the base link
    if translation is None:
        if mesh.is_watertight:
            translation = mesh.center_mass
        else:
            translation = mesh.centroid

    # Rotate the object around the CoM of the base link frame by the transpose of its canonical orientation
    # so that the object has identity orientation in the end
    rotation_matrix = np.eye(4)
    rotation_matrix[:3, :3] = R.from_quat(rotation).as_matrix().T
    mesh.apply_transform(trimesh.transformations.translation_matrix(-translation))
    mesh.apply_transform(rotation_matrix)
    mesh.apply_transform(trimesh.transformations.translation_matrix(translation))


def get_base_link_center(mesh):
    mesh_copy = mesh.copy()

    # Correct coordinate system change in pybullet
    coordinate_matrix = np.eye(4)
    coordinate_matrix[:3, :3] = R.from_euler("xyz", [np.pi / 2, 0, 0]).as_matrix()

    # Original mesh 1 unit = 1mm
    scale_matrix = trimesh.transformations.scale_matrix(0.001)

    mesh_copy.apply_transform(scale_matrix).apply_transform(coordinate_matrix)
    return get_mesh_center(mesh_copy)


def get_mesh_center(mesh):
    if mesh.is_watertight:
        return mesh.center_mass
    else:
        return mesh.centroid


def main():
    p.connect(p.DIRECT)
    parent_to_children = build_object_hierarchy()

    scene_urdf_tree = ET.parse("template.urdf")
    scene_tree_root = scene_urdf_tree.getroot()
    scene_tree_root.attrib = {"name": "igibson_scene"}
    world_link = ET.SubElement(scene_tree_root, "link")
    world_link.attrib = {"name": "world"}

    for obj_inst in parent_to_children:
        urdf_tree = ET.parse("template.urdf")
        tree_root = urdf_tree.getroot()

        obj_inst_name = "-".join([obj_inst[0], obj_inst[1]])
        category = "test_" + obj_inst[0]
        processed_obj_inst_folder = os.path.join(processed_folder, category, obj_inst_name)
        os.makedirs(processed_obj_inst_folder, exist_ok=True)

        obj_parent_to_children = parent_to_children[obj_inst]
        parent_sets = ["root"]
        parent_centers = {}
        base_link_center = None
        while len(parent_sets) > 0:
            next_parent_sets = []
            for parent_link_name in parent_sets:
                if parent_link_name not in obj_parent_to_children["hierarchy"]:
                    continue
                for link_name, joint_type in obj_parent_to_children["hierarchy"][parent_link_name]:
                    # It's guaranteed that the first link_name will be the base link
                    # print("obj_inst", obj_inst_name, "link_name", link_name)
                    next_parent_sets.append(link_name)
                    if joint_type is None:
                        obj_name = "-".join([obj_inst[0], obj_inst[1], link_name])
                    else:
                        obj_name = "-".join(
                            [obj_inst[0], obj_inst[1], link_name, parent_link_name, joint_type, "lower"]
                        )

                    obj_dir = os.path.join(root_folder, obj_name)
                    # Fix MTL file path
                    for mtl_file in os.listdir(obj_dir):
                        if mtl_file.endswith(".mtl"):
                            fix_mtl_file(obj_dir, mtl_file)

                    obj_file = os.path.join(obj_dir, "{}.obj".format(obj_name))
                    mesh = trimesh.load(obj_file, process=False)

                    # Set canonical_orientation by querying the metadata file of the base link
                    if joint_type is None:
                        json_file = os.path.join(obj_dir, "{}.json".format(obj_name))
                        if os.path.isfile(json_file):
                            with open(json_file) as f:
                                metadata = json.load(f)
                            canonical_orientation = np.array(metadata["orientation"])
                        else:
                            canonical_orientation = np.array([0.0, 0.0, 0.0, 1.0])
                        base_link_center = get_base_link_center(mesh)

                    # base_link_center will be None when mesh is the base link
                    scale_rotate_mesh(mesh, base_link_center, canonical_orientation)
                    center = get_mesh_center(mesh)

                    parent_centers[link_name] = center

                    # Cache "lower" mesh before translation
                    lower_mesh = mesh.copy()

                    # Make the mesh centered at its CoM
                    mesh.apply_translation(-center)

                    # Somehow we need to manually write the vertex normals to cache
                    mesh._cache.cache["vertex_normals"] = mesh.vertex_normals

                    # Save the mesh
                    obj_link_folder = os.path.join(processed_obj_inst_folder, obj_name)
                    os.makedirs(obj_link_folder, exist_ok=True)
                    obj_relative_path = "{}.obj".format(obj_name)
                    obj_path = os.path.join(obj_link_folder, obj_relative_path)
                    mesh.export(obj_path, file_type="obj")

                    # Move the mesh to the correct path
                    obj_link_mesh_folder = os.path.join(processed_obj_inst_folder, "shape")
                    os.makedirs(obj_link_mesh_folder, exist_ok=True)
                    obj_link_visual_mesh_folder = os.path.join(obj_link_mesh_folder, "visual")
                    os.makedirs(obj_link_visual_mesh_folder, exist_ok=True)
                    obj_link_collision_mesh_folder = os.path.join(obj_link_mesh_folder, "collision")
                    os.makedirs(obj_link_collision_mesh_folder, exist_ok=True)
                    obj_link_material_folder = os.path.join(processed_obj_inst_folder, "material")
                    os.makedirs(obj_link_material_folder, exist_ok=True)

                    src_mtl_file = os.path.join(obj_link_folder, "material_0.mtl")
                    dst_mtl_file = os.path.join(obj_link_visual_mesh_folder, "{}.mtl".format(obj_name))
                    shutil.copy(src_mtl_file, dst_mtl_file)
                    src_texture_file = os.path.join(obj_link_folder, "material_0.png")
                    dst_texture_file = os.path.join(obj_link_material_folder, "{}.png".format(obj_name))
                    shutil.copy(src_texture_file, dst_texture_file)
                    src_obj_file = obj_path
                    visual_shape_file = os.path.join(obj_link_visual_mesh_folder, obj_relative_path)
                    shutil.copy(src_obj_file, visual_shape_file)

                    # Generate collision mesh
                    collision_shape_file = os.path.join(obj_link_collision_mesh_folder, obj_relative_path)
                    cmd = '../../blender_utils/vhacd --input "{}" --output "{}"'.format(
                        visual_shape_file, collision_shape_file
                    )
                    subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

                    # Modify MTL reference in OBJ file
                    with open(visual_shape_file, "r") as f:
                        new_lines = []
                        for line in f.readlines():
                            if "mtllib material_0.mtl" in line:
                                line = "mtllib {}.mtl\n".format(obj_name)
                            new_lines.append(line)

                    with open(visual_shape_file, "w") as f:
                        for line in new_lines:
                            f.write(line)

                    # Modify texture reference in MTL file
                    with open(dst_mtl_file, "r") as f:
                        new_lines = []
                        for line in f.readlines():
                            if "map_Kd material_0.png" in line:
                                line = "map_Kd ../../material/{}.png\n".format(obj_name)
                            new_lines.append(line)

                    with open(dst_mtl_file, "w") as f:
                        for line in new_lines:
                            f.write(line)

                    # Create the link in URDF
                    tree_root.attrib = {"name": obj_inst_name}
                    link = ET.SubElement(tree_root, "link")
                    link.attrib = {"name": link_name}
                    visual = ET.SubElement(link, "visual")
                    visual_origin = ET.SubElement(visual, "origin")
                    visual_origin.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
                    visual_geometry = ET.SubElement(visual, "geometry")
                    visual_mesh = ET.SubElement(visual_geometry, "mesh")
                    visual_mesh.attrib = {"filename": os.path.join("shape", "visual", obj_relative_path)}

                    collision = ET.SubElement(link, "collision")
                    collision_origin = ET.SubElement(collision, "origin")
                    collision_origin.attrib = {"xyz": " ".join([str(item) for item in [0.0] * 3])}
                    collision_geometry = ET.SubElement(collision, "geometry")
                    collision_mesh = ET.SubElement(collision_geometry, "mesh")
                    collision_mesh.attrib = {"filename": os.path.join("shape", "collision", obj_relative_path)}

                    if joint_type is not None:
                        upper_obj_dir = obj_dir.replace("lower", "upper")
                        # Fix the MTL file path
                        for mtl_file in os.listdir(upper_obj_dir):
                            if mtl_file.endswith(".mtl"):
                                fix_mtl_file(upper_obj_dir, mtl_file)

                        # Find the upper joint limit mesh
                        upper_mesh_file = obj_file.replace("lower", "upper")
                        upper_mesh = trimesh.load(upper_mesh_file, process=False)
                        scale_rotate_mesh(upper_mesh, base_link_center, canonical_orientation)

                        # Find the center of the parent link
                        parent_center = parent_centers[parent_link_name]

                        if joint_type == "r":
                            # Revolute joint
                            num_v = lower_mesh.vertices.shape[0]
                            random_index = np.random.choice(num_v, min(num_v, 20), replace=False)
                            from_vertices = lower_mesh.vertices[random_index]
                            to_vertices = upper_mesh.vertices[random_index]

                            # Find joint axis and joint limit
                            r = R.align_vectors(
                                to_vertices - np.mean(to_vertices, axis=0),
                                from_vertices - np.mean(from_vertices, axis=0),
                            )[0]
                            upper_limit = r.magnitude()
                            joint_axis_xyz = r.as_rotvec() / r.magnitude()

                            # Find the translation part of the joint origin
                            r_mat = r.as_matrix()
                            from_vertices_mean = from_vertices.mean(axis=0)
                            to_vertices_mean = to_vertices.mean(axis=0)
                            left_mat = np.eye(3) - r_mat
                            t = np.linalg.lstsq(
                                left_mat, (to_vertices_mean - np.dot(r_mat, from_vertices_mean)), rcond=None
                            )[0]

                            # Find the translation part of the joint origin that is closest to the CoM of the link
                            # The joint origin has infinite number of solutions along the joint axis
                            a = t
                            b = t + joint_axis_xyz
                            pt = center

                            ap = pt - a
                            ab = b - a
                            joint_origin_xyz = a + np.dot(ap, ab) / np.dot(ab, ab) * ab

                            # Assign visual and collision mesh orogin based on the diff between CoM and joint origin
                            mesh_offset = center - joint_origin_xyz
                            visual_origin.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}
                            collision_origin.attrib = {"xyz": " ".join([str(item) for item in mesh_offset])}

                            # Assign the joint origin relative to the parent CoM
                            joint_origin_xyz = joint_origin_xyz - parent_center
                        else:
                            # Prismatic joint
                            diff = upper_mesh.centroid - lower_mesh.centroid

                            # Find joint axis and joint limit
                            upper_limit = np.linalg.norm(diff)
                            joint_axis_xyz = diff / upper_limit

                            # Assign the joint origin relative to the parent CoM
                            joint_origin_xyz = center - parent_center

                        # Create the joint in the URDF
                        joint = ET.SubElement(tree_root, "joint")
                        joint.attrib = {
                            "name": "j_{}".format(link_name),
                            "type": "revolute" if joint_type == "r" else "prismatic",
                        }
                        joint_origin = ET.SubElement(joint, "origin")
                        joint_origin.attrib = {"xyz": " ".join([str(item) for item in joint_origin_xyz])}
                        joint_axis = ET.SubElement(joint, "axis")
                        joint_axis.attrib = {"xyz": " ".join([str(item) for item in joint_axis_xyz])}
                        joint_parent = ET.SubElement(joint, "parent")
                        joint_parent.attrib = {"link": parent_link_name}
                        joint_child = ET.SubElement(joint, "child")
                        joint_child.attrib = {"link": link_name}
                        joint_limit = ET.SubElement(joint, "limit")
                        joint_limit.attrib = {"lower": str(0.0), "upper": str(upper_limit)}

            parent_sets = next_parent_sets

        urdf_path = os.path.join(processed_obj_inst_folder, "{}.urdf".format(obj_inst_name))
        xmlstr = minidom.parseString(ET.tostring(tree_root)).toprettyxml(indent="   ")
        with open(urdf_path, "w") as f:
            f.write(xmlstr)
        tree = ET.parse(urdf_path)
        print(urdf_path)
        tree.write(urdf_path, xml_declaration=True)

        # Save medadata json
        body_id = p.loadURDF(urdf_path)
        lower, upper = p.getAABB(body_id)
        base_link_offset = ((np.array(lower) + np.array(upper)) / 2.0).tolist()
        bbox_size = (np.array(upper) - np.array(lower)).tolist()
        metadata = {
            "base_link_offset": base_link_offset,
            "bbox_size": bbox_size,
        }
        obj_misc_folder = os.path.join(processed_obj_inst_folder, "misc")
        os.makedirs(obj_misc_folder, exist_ok=True)
        metadata_file = os.path.join(obj_misc_folder, "metadata.json")
        with open(metadata_file, "w+") as f:
            json.dump(metadata, f)

        rotated_offset = p.multiplyTransforms(
            [0, 0, 0], canonical_orientation, -np.array(base_link_offset), [0, 0, 0, 1]
        )[0]
        bbox_center = base_link_center - rotated_offset

        # Save pose to scene URDF
        scene_link = ET.SubElement(scene_tree_root, "link")
        obj_model_name = obj_inst_name
        obj_link_name = obj_inst_name
        scene_link.attrib = {
            "bounding_box": " ".join([str(item) for item in bbox_size]),
            "category": category,
            "model": obj_model_name,
            "name": obj_link_name,
        }
        joint = ET.SubElement(scene_tree_root, "joint")
        joint.attrib = {
            "name": "j_{}".format(obj_link_name),
            "type": "fixed",
        }
        joint_origin = ET.SubElement(joint, "origin")
        joint_origin_xyz = bbox_center.tolist()
        joint_origin_rpy = p.getEulerFromQuaternion(canonical_orientation)
        joint_origin.attrib = {
            "xyz": " ".join([str(item) for item in joint_origin_xyz]),
            "rpy": " ".join([str(item) for item in joint_origin_rpy]),
        }
        joint_parent = ET.SubElement(joint, "parent")
        joint_parent.attrib = {"link": "world"}
        joint_child = ET.SubElement(joint, "child")
        joint_child.attrib = {"link": obj_link_name}

    scene_urdf_dir = os.path.join(scene_processed_folder, scene_name, "urdf")
    os.makedirs(scene_urdf_dir, exist_ok=True)
    scene_urdf_file = os.path.join(scene_urdf_dir, "{}_best.urdf".format(scene_name))
    xmlstr = minidom.parseString(ET.tostring(scene_tree_root)).toprettyxml(indent="   ")
    with open(scene_urdf_file, "w") as f:
        f.write(xmlstr)
    tree = ET.parse(scene_urdf_file)
    print(scene_urdf_file)
    tree.write(scene_urdf_file, xml_declaration=True)


if __name__ == "__main__":
    main()
