import os

import numpy as np
import open3d as o3d
import pandas as pd

import igibson

base_dir = os.path.join(igibson.ig_dataset_path, "objects")

object_models = []
categories = os.listdir(base_dir)
for category in categories:
    if "json" not in category:
        objs = os.listdir(os.path.join(base_dir, category))
        for obj_id in objs:
            obj_model_dir = base_dir + "/{}/{}/shape/collision".format(category, obj_id)
            if os.path.exists(obj_model_dir):
                model_files = [
                    os.path.join(obj_model_dir, obj)
                    for obj in os.listdir(obj_model_dir)
                    if obj.endswith(".obj") and "original.obj" not in obj
                ]
                object_models.append(
                    {
                        "category": category,
                        "object_id": obj_id,
                        "model_files": model_files,
                    }
                )
            else:
                print("Warning: collision mesh missing for {}/{}".format(category, obj_id))

vals = []
names = []

original_vertices_ls = []
original_triangles_ls = []

compressed_vertices_ls = []
compressed_triangles_ls = []

for obj in object_models:
    original_vertices = 0
    original_triangles = 0

    compressed_vertices = 0
    compressed_triangles = 0

    for model_file in obj["model_files"]:
        original_mesh = o3d.io.read_triangle_mesh(os.path.splitext(model_file)[0] + "_original.obj")
        mesh = o3d.io.read_triangle_mesh(str(model_file))
        original_vertices += np.asarray(original_mesh.vertices).shape[0]
        original_triangles += np.asarray(original_mesh.triangles).shape[0]
        compressed_vertices += np.asarray(mesh.vertices).shape[0]
        compressed_triangles += np.asarray(mesh.triangles).shape[0]

    original_vertices_ls.append(original_vertices)
    original_triangles_ls.append(original_triangles)

    compressed_vertices_ls.append(compressed_vertices)
    compressed_triangles_ls.append(compressed_triangles)
    names.append("{}/{}".format(obj["category"], obj["object_id"]))

    vals.append(
        [
            "{}/{}".format(obj["category"], obj["object_id"]),
            compressed_vertices,
            original_vertices,
        ]
    )

df = pd.DataFrame(
    {
        "model_names": names,
        "original_vertices": original_vertices_ls,
        "original_triangles": original_triangles_ls,
        "compressed_vertices": compressed_vertices_ls,
        "compressed_triangles": compressed_triangles_ls,
    }
)
df = df.sort_values("original_vertices")
print("Sorted by original vertices")
print(df)

df.to_csv("model_qc.csv")

df = df.sort_values("compressed_vertices")
print("Sorted by compressed vertices")
print(df)
