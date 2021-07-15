import csv
import os
import subprocess
from shutil import copyfile

import igibson

base_dir = os.path.join(igibson.ig_dataset_path, "objects")

objects = []
categories = os.listdir(base_dir)
for category in categories:
    if "json" not in category:
        objs = os.listdir(os.path.join(base_dir, category))
        for obj in objs:
            objects.append({"category": category, "object_id": obj})

mesh_names = []
successes = []
mesh_counts = []

for obj in objects:
    category = obj["category"]
    object_id = obj["object_id"]

    object_export_dir = base_dir + "/{}/{}/shape".format(category, object_id)
    input_dir = os.path.join(object_export_dir, "collision")

    if not os.path.exists(input_dir) or len(os.listdir(input_dir)) == 0:
        print("Inputs dir {} does not exist, or is missing mesh files.".format(input_dir))
        continue

    success = True
    mesh_count = 0
    meshes = [mesh for mesh in os.listdir(input_dir) if not mesh.endswith("_original.obj")]
    # if category != "swivel_chair" or object_id != "2627":
    #     continue
    for mesh in meshes:
        # Copy the original mesh to a backup file
        backup_mesh = os.path.join(input_dir, os.path.splitext(mesh)[0] + "_original.obj")
        input_mesh = os.path.join(input_dir, mesh)

        if not os.path.exists(backup_mesh):
            copyfile(input_mesh, backup_mesh)
        try:
            vhacd_path = os.path.join(igibson.root_path, "utils", "data_utils", "blender_utils", "vhacd")
            # Note, probably should parse the output to make sure this was actually succesful
            pipe = subprocess.run([vhacd_path, "--input", backup_mesh, "--output", input_mesh])
            mesh_count += 1
        except:
            print("couldn't process {}".format(input_dir))
            success = False

    mesh_names.append("{}/{}".format(category, object_id))
    successes.append(success)
    mesh_counts.append(mesh_count)

with open("vhacd_manifest.csv", "w") as f:
    writer = csv.writer(f)
    writer.writerow(["video_id", "category", "object_id"])
    for mesh_name, success, mesh_count in zip(mesh_names, successes, mesh_counts):
        writer.writerow([mesh_name, success, mesh_count])
