import itertools
import json
import os
import xml.etree.ElementTree as ET

import numpy as np
import pybullet as p
import tqdm
import trimesh

import igibson

SKIP_EXISTING = True
IGNORE_ERRORS = True


def get_categories():
    dir = os.path.join(igibson.ig_dataset_path, "objects")
    return [cat for cat in os.listdir(dir) if os.path.isdir(get_category_directory(cat))]


def get_category_directory(category):
    return os.path.join(igibson.ig_dataset_path, "objects", category)


def get_urdf_filename(folder):
    return os.path.join(folder, os.path.basename(folder) + ".urdf")


def get_metadata_filename(objdir):
    return os.path.join(objdir, "misc", "metadata.json")


def get_corner_positions(base, rotation, size):
    quat = p.getQuaternionFromEuler(rotation)
    options = [-1, 1]
    outputs = []
    for pos in itertools.product(options, options, options):
        res = p.multiplyTransforms(base, quat, np.array(pos) * size / 2.0, [0, 0, 0, 1])
        outputs.append(res)
    return outputs


def main():
    # Collect the relevant categories.
    categories = get_categories()
    print("%d categories: %s" % (len(categories), ", ".join(categories)))

    # Now collect the actual objects.
    objects = []
    for cat in categories:
        cd = get_category_directory(cat)
        for objdir in os.listdir(cd):
            objdirfull = os.path.join(cd, objdir)
            objects.append(objdirfull)

    objects.sort()
    print("%d objects.\n" % len(objects))

    # Filter out already-processed objects.
    objects_to_process = []
    for objdirfull in objects:
        mfn = get_metadata_filename(objdirfull)
        with open(mfn, "r") as mf:
            meta = json.load(mf)

        if "link_bounding_boxes" in meta and len(meta["link_bounding_boxes"]) > 0:
            if SKIP_EXISTING:
                continue

        objects_to_process.append(objdirfull)

    # Now process the remaining.
    for objdirfull in tqdm.tqdm(objects_to_process):
        mfn = get_metadata_filename(objdirfull)
        with open(mfn, "r") as mf:
            meta = json.load(mf)

        try:
            urdf_file = get_urdf_filename(objdirfull)
            tree = ET.parse(urdf_file)
            links = tree.findall("link")

            link_bounding_boxes = {}
            for link in links:
                this_link_bounding_boxes = {}

                for mesh_type in ["collision", "visual"]:
                    try:
                        # For each type of mesh, we first combine all of the different mesh files into a single mesh.
                        mesh_type_nodes = link.findall(mesh_type)
                        combined_mesh = trimesh.Trimesh()
                        mesh_count = 0

                        for mesh_type_node in mesh_type_nodes:
                            origin = mesh_type_node.find("origin")
                            rotation = origin.attrib["rpy"] if "rpy" in origin.attrib else "0 0 0"
                            rotation = np.array([float(x) for x in rotation.split(" ")])
                            assert rotation.shape == (3,)
                            translation = origin.attrib["xyz"] if "xyz" in origin.attrib else "0 0 0"
                            translation = np.array([float(x) for x in translation.split(" ")])
                            assert translation.shape == (3,)
                            mesh_nodes = mesh_type_node.findall("geometry/mesh")

                            for mesh_node in mesh_nodes:
                                object_file = os.path.join(objdirfull, mesh_node.attrib["filename"])
                                mesh = trimesh.load(object_file, force="mesh")
                                scale = mesh_node.attrib["scale"] if "scale" in mesh_node.attrib else "1 1 1"
                                scale = np.array([float(x) for x in scale.split(" ")])

                                # Apply the translation & scaling and add the mesh onto the combined mesh.
                                combined_mesh += mesh.apply_transform(
                                    trimesh.transformations.compose_matrix(
                                        scale=scale, angles=rotation, translate=translation
                                    )
                                )

                                if "scale" in mesh_node.attrib:
                                    mesh_count += 1

                        # Now that we have the combined mesh, let's simply compute the bounding box.
                        bbox = combined_mesh.bounding_box
                        bbox = {
                            "extent": np.array(bbox.primitive.extents).tolist(),
                            "transform": np.array(bbox.primitive.transform).tolist(),
                        }
                        bbox_oriented = combined_mesh.bounding_box_oriented
                        bbox_oriented = {
                            "extent": np.array(bbox_oriented.primitive.extents).tolist(),
                            "transform": np.array(bbox_oriented.primitive.transform).tolist(),
                        }

                        if mesh_count > 1:
                            combined_mesh.show()

                        this_link_bounding_boxes[mesh_type] = {"axis_aligned": bbox, "oriented": bbox_oriented}
                    except:
                        print(
                            "Problem with %s mesh in link %s in obj %s" % (mesh_type, link.attrib["name"], objdirfull)
                        )

                if len(this_link_bounding_boxes) > 0:
                    link_bounding_boxes[link.attrib["name"]] = this_link_bounding_boxes

            if len(link_bounding_boxes) == 0:
                raise ValueError("No links have a mesh for object %s" % objdirfull)

            meta["link_bounding_boxes"] = link_bounding_boxes

            with open(mfn, "w") as mf:
                json.dump(meta, mf)
        except Exception as e:
            combined_mesh.show()

            if IGNORE_ERRORS:
                print("Something went wrong with ", objdirfull)
            else:
                raise


if __name__ == "__main__":
    main()
