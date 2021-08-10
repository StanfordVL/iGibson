"""
    This file needs to be run in blender, and requires a DISPLAY (virtualgl may work):
    Requires the following on your path:
        * blender 2.9.x
        * ffmpeg
    Run as:
        blender -b --python mesh_decimation_visualization.py -- --ig_dataset $(pwd)/../../../data/ig_dataset
"""
import argparse
import csv
import math
import os
import subprocess
import sys

import bpy


def collision_filter(x):
    if not x.endswith("original.obj") and x.endswith(".obj"):
        return True
    else:
        return False


def visual_filter(x):
    if x.endswith(".obj"):
        return True
    else:
        return False


def import_and_join_obj(meshes):
    # Import all of the visual meshes
    for mesh in meshes:
        bpy.ops.import_scene.obj(filepath=mesh)

    # Deselect all
    bpy.ops.object.select_all(action="DESELECT")

    # Select all imported meshes
    mesh_objects = [m for m in bpy.context.scene.objects if m.type == "MESH"]

    for objects in mesh_objects:
        objects.select_set(state=True)
        bpy.context.view_layer.objects.active = objects

    # Join all imported meshes into a single object
    bpy.ops.object.join()


def render_frame(
    visual_meshes,
    collision_meshes,
    coll_tmpdir="collision/tmp",
    vis_tmpdir="visual/tmp",
):
    # Delete all scene objects except light
    scene = bpy.data.scenes[0]
    for o in bpy.context.scene.objects:
        if o.type != "LIGHT":
            o.select_set(True)
    bpy.ops.object.delete()

    # Import and join the meshes
    import_and_join_obj(visual_meshes)

    # Get the current scene handle
    scn = bpy.context.scene

    # Set the target for the camera to the joined object
    target = bpy.context.selected_objects[0]
    bpy.context.view_layer.objects.active = target

    # Center the origin on mesh
    bpy.ops.object.origin_set(type="ORIGIN_GEOMETRY")

    cam_x_pos = max([v[0] for v in target.bound_box]) * 5
    cam_y_pos = max([v[1] for v in target.bound_box]) * 5
    cam_z_pos = max([v[2] for v in target.bound_box]) * 5

    # Set the rotational center to the active object
    rot_centre = bpy.data.objects.new("rot_centre", None)
    bpy.context.collection.objects.link(rot_centre)
    rot_centre.location = target.location

    # Create a camera and link it to the active rotational center object
    camera = bpy.data.objects.new("camera", bpy.data.cameras.new("camera"))
    scn.camera = camera
    bpy.context.scene.collection.objects.link(camera)

    # Set the camera location
    camera.location = (cam_x_pos, cam_y_pos, cam_z_pos)
    camera.parent = rot_centre

    # Track the camera to the rotational center
    m = camera.constraints.new("TRACK_TO")
    m.target = target
    m.track_axis = "TRACK_NEGATIVE_Z"
    m.up_axis = "UP_Y"

    # Set up the rotational center to rotate
    rot_centre.rotation_euler.z = 0.0
    rot_centre.keyframe_insert("rotation_euler", index=2, frame=1)
    rot_centre.rotation_euler.z = math.radians(360.0)
    rot_centre.keyframe_insert("rotation_euler", index=2, frame=101)

    # Linearly interpolate the rotation across 100 frames
    for c in rot_centre.animation_data.action.fcurves:
        for k in c.keyframe_points:
            k.interpolation = "LINEAR"

    scn.frame_end = 100

    # Set the output render path to a temporary file
    scene.render.filepath = vis_tmpdir

    # Render the animation to the file
    bpy.ops.render.render(animation=True)

    # Delete the mesh but keep the camera
    for o in bpy.context.scene.objects:
        if o.type == "MESH":
            o.select_set(True)
    bpy.ops.object.delete()

    # Import and join the meshes
    import_and_join_obj(collision_meshes)

    target = bpy.context.collection.objects["rot_centre"]
    bpy.context.view_layer.objects.active = target
    m.target = target

    # Set the output render path to a temporary file
    scene.render.filepath = coll_tmpdir

    # Render the animation to the file
    bpy.ops.render.render(animation=True)


def main():

    argv = sys.argv
    if "--" not in argv:
        argv = []  # as if no args are passed
    else:
        argv = argv[argv.index("--") + 1 :]  # get all args after "--"

    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument(
        "--ig_dataset",
        required=True,
        type=str,
        help="path to ig_dataset (typically /path/to/iGibson/igibson/data/ig_dataset)",
    )
    args = parser.parse_args(argv)
    basedir = os.path.join(args.ig_dataset, "objects")

    # Set up scene for eevee, this is faster than cycles with CUDA
    scene = bpy.data.scenes[0]
    scene.render.engine = "BLENDER_EEVEE"
    scene.render.resolution_x = 720
    scene.render.resolution_y = 720
    scene.render.resolution_percentage = 50
    scene.eevee.volumetric_samples = 2
    scene.eevee.taa_render_samples = 2

    # Make a list of all object categories/ids
    objects = []
    categories = os.listdir(basedir)
    for category in categories:
        if "json" not in category:
            objs = os.listdir(os.path.join(basedir, category))
            for obj in objs:
                objects.append({"category": category, "object_id": obj})

    # This will store the metadata matching categories, object_ids, and video ids
    categories = []
    object_ids = []
    video_ids = []

    # Generate the videos across the dataset
    for idx, obj in enumerate(objects):
        category = obj["category"]
        object_id = obj["object_id"]
        categories.append(obj["category"])
        object_ids.append(obj["object_id"])
        video_ids.append(idx)

        object_export_dir = os.path.join(basedir, category, object_id, "shape")
        visual_dir = os.path.join(object_export_dir, "visual")
        collision_dir = os.path.join(object_export_dir, "collision")

        if not os.path.exists(visual_dir) or len(os.listdir(visual_dir)) == 0:
            continue
        if not os.path.exists(collision_dir) or len(os.listdir(collision_dir)) == 0:
            continue

        visual_meshes = [item for item in os.listdir(visual_dir) if visual_filter(item)]
        visual_mesh_paths = []
        for mesh in visual_meshes:
            visual_mesh_paths.append(os.path.join(visual_dir, mesh))

        collision_meshes = [item for item in os.listdir(collision_dir) if collision_filter(item)]
        collision_mesh_paths = []
        for mesh in collision_meshes:
            collision_mesh_paths.append(os.path.join(collision_dir, mesh))

        # Make the output directory
        if not os.path.exists("final_videos"):
            os.mkdir("final_videos")

        # Render each set of frames into a video, and then process this into a side by side video
        try:
            render_frame(visual_meshes=visual_mesh_paths, collision_meshes=collision_mesh_paths)
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    "visual/tmp%04d.png",
                    "-c:v",
                    "libx264",
                    "-vf",
                    "fps=25",
                    "-pix_fmt",
                    "yuv420p",
                    "visual/visual.mp4",
                ],
                capture_output=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    "collision/tmp%04d.png",
                    "-c:v",
                    "libx264",
                    "-vf",
                    "fps=25",
                    "-pix_fmt",
                    "yuv420p",
                    "collision/collision.mp4",
                ],
                capture_output=True,
            )
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i",
                    "visual/visual.mp4",
                    "-i",
                    "collision/collision.mp4",
                    "-filter_complex",
                    "hstack",
                    "final_videos/{}.mp4".format(idx),
                ],
                capture_output=True,
            )
        except:
            print("couldn't process {}".format(visual_dir))

        # Control blender's memory usage
        if idx % 10 == 0:
            bpy.ops.outliner.orphans_purge(do_recursive=True)

    with open("final_videos/manifest.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerow(["video_id", "category", "object_id"])
        for video_id, category, object_id in zip(video_ids, categories, video_ids):
            writer.writerow([video_id, category, object_id])


main()
