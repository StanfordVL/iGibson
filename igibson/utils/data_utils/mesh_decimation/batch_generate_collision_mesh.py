import os
import subprocess

import igibson

objects = []
base_dir = os.path.join(igibson.ig_dataset_path, "objects")
categories = os.listdir(base_dir)
for category in categories:
    if "json" not in category:
        objs = os.listdir(os.path.join(base_dir, category))
        for obj in objs:
            objects.append({"category": category, "object_id": obj})

for obj in objects:
    category = obj["category"]
    object_id = obj["object_id"]

    object_export_dir = base_dir + "/{}/{}/shape".format(category, object_id)
    urdf = os.path.join(base_dir, category, object_id, "{}.urdf".format(object_id))
    input_dir = os.path.join(object_export_dir, "collision", "decimated")
    if os.path.exists(input_dir):
        output_dir = os.path.join(object_export_dir, "collision")
        collision_mesh_script = os.path.join(
            igibson.root_path, "utils", "data_utils", "ext_object", "escripts", "step_2_collision_mesh.py"
        )
        pipe = subprocess.run(
            [
                "python",
                collision_mesh_script,
                "--input_dir",
                input_dir,
                "--output_dir",
                output_dir,
                "--object_name",
                object_id,
                "--urdf",
                urdf,
            ]
        )
