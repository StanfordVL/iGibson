import os
import sys

import bpy

script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, "../../blender_utils")
sys.path.append(utils_dir)

from utils import bake_model, clean_unused, export_ig_object, import_obj_folder

"""
to run it:
blender -b --python step_1_visual_mesh_multi_uv.py -- --bake --up Z --forward X --source_blend_file {PATH_TO_YOUR_OBJECT} --dest_dir {PATH_TO_IGIBSON_ASSET}/objects/{OBJECT_CATEGORY}/{OBJECT_NAME}
"""


#############################################
# Parse command line arguments
#############################################


def get_arg(argv, flag, default=None):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return default


should_bake = "--bake" in sys.argv

axis = ["X", "Y", "Z", "-X", "-Y", "-Z"]
import_axis_up = get_arg(sys.argv, "--up", default="Z")
if import_axis_up not in axis:
    raise ValueError("Axis up not supported: {} (should be among X,Y,Z,-X,-Y,-Z)".format(import_axis_up))

import_axis_forward = get_arg(sys.argv, "--forward", default="X")
if import_axis_forward not in axis:
    raise ValueError("Axis forward not supported: {} (should be among X,Y,Z,-X,-Y,-Z)".format(import_axis_forward))

# Whether to merge all the visual meshes into a single mesh
merge_obj = get_arg(sys.argv, "--merge_obj", default="1")
merge_obj = int(merge_obj)
assert merge_obj in [0, 1]

source_blend_file = get_arg(sys.argv, "--source_blend_file")
source_dir = get_arg(sys.argv, "--source_dir")
if source_blend_file is None and source_dir is None:
    raise ValueError("Source directory and source blend file are both empty.")
dest_dir = get_arg(sys.argv, "--dest_dir")
if dest_dir is None:
    raise ValueError("Destination directory not specified.")
os.makedirs(dest_dir, exist_ok=True)

#############################################
# Open source blend file or import obj files from source dir
#############################################
if source_blend_file is not None:
    # open source blend file
    bpy.ops.wm.open_mainfile(filepath=source_blend_file)

else:
    # import obj files from source dir
    for on in bpy.context.scene.objects.keys():
        obj = bpy.context.scene.objects[on]
        bpy.data.objects.remove(obj)
    clean_unused()

    model_id = os.path.basename(source_dir)
    import_obj_folder(model_id, source_dir, up=import_axis_up, forward=import_axis_forward)


#############################################
# Decimate the mesh
#############################################
max_num_verts = 10000

meshes = []
for obj in bpy.data.objects:
    if obj.type == "MESH":
        meshes.append(obj)
total_num_verts = sum([len(mesh.data.vertices) for mesh in meshes])
print("total_num_verts (before decimation)", total_num_verts)
print("max_num_verts", max_num_verts)
if total_num_verts > max_num_verts:
    ratio = float(max_num_verts) / total_num_verts
    bpy.ops.object.select_all(action="SELECT")
    bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.decimate(ratio=ratio)
    bpy.ops.object.mode_set(mode="OBJECT")
    meshes = []
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            meshes.append(obj)
    total_num_verts = sum([len(mesh.data.vertices) for mesh in meshes])
    print("decimation ratio", ratio)
    print("total_num_verts (after decimation)", total_num_verts)

#############################################
# Optional UV Unwrapping
# This only needed if baking will be performed
#############################################
if should_bake:
    uv_unwrapped = False
    # for o in bpy.context.scene.objects:
    #    if not o.data.uv_layers:
    #        uv_unwrapped = False
    if not uv_unwrapped:
        bpy.ops.object.select_all(action="SELECT")
        bpy.context.view_layer.objects.active = bpy.context.selected_objects[0]
        bpy.ops.object.mode_set(mode="OBJECT")
        vl = bpy.context.view_layer
        bpy.ops.object.select_all(action="DESELECT")
        for on in bpy.context.scene.objects.keys():
            obj = bpy.context.scene.objects[on]
            new_uv = bpy.context.scene.objects[on].data.uv_layers.new(name="obj_uv")
            new_uv.active = True
            vl.objects.active = obj
            obj.select_set(True)
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(
            angle_limit=1.15192, island_margin=0.002, area_weight=0.0, correct_aspect=True, scale_to_bounds=False
        )
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode="OBJECT")


#############################################
# Detect glass
#############################################
has_glass = False
for i in range(len(bpy.data.materials)):
    print(bpy.data.materials[i].name.lower())
    if "glass" in bpy.data.materials[i].name.lower():
        has_glass = True
        # sometimes the glass effect is achieved by Metallic=1.0
        # change it to Transmission=1.0
        if "Principled BSDF" in bpy.data.materials[i].node_tree.nodes:
            principled_bsdf = bpy.data.materials[i].node_tree.nodes["Principled BSDF"]
            principled_bsdf.inputs["Transmission"].default_value = 1.0
            principled_bsdf.inputs["Metallic"].default_value = 0.0


#############################################
# Export the un-merged model
#############################################
if not merge_obj:
    # Keep the original import_axis up and forward. Otherwise, the original
    # URDF won't work (e.g joint axes will be incorrect)
    export_ig_object(dest_dir, save_material=not should_bake, up=import_axis_up, forward=import_axis_forward)

#############################################
# Optional Texture Baking
#############################################
if should_bake:
    mat_dir = os.path.join(dest_dir, "material")
    os.makedirs(mat_dir, exist_ok=True)

    # bpy.ops.wm.open_mainfile(filepath=blend_path)
    # import_ig_object(model_root, import_mat=True)
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.join()

    resolution = 1024
    channels = {
        "DIFFUSE": (resolution * 2, 32),
        "ROUGHNESS": (resolution, 16),
        "METALLIC": (resolution, 16),
        "NORMAL": (resolution, 16),
    }
    if has_glass:
        channels["TRANSMISSION"] = (resolution * 2, 32)
        # add world light
        world = bpy.context.scene.world
        if world is None:
            world = bpy.data.worlds.new("World")
        world.use_nodes = True
        # changing these values does affect the render.
        bg = world.node_tree.nodes["Background"]
        bg.inputs[0].default_value[:3] = (1, 1, 1)
        bg.inputs[1].default_value = 1.0

    bake_model(mat_dir, channels, overwrite=True, add_uv_node=True)


#############################################
# Export the merged model
#############################################
if merge_obj:
    export_ig_object(dest_dir, save_material=not should_bake)

# # optionally save blend file
# if source_blend_file is not None:
#     output_blend_file = os.path.join(os.path.dirname(source_blend_file), 'processed.blend')
# else:
#     output_blend_file = os.path.join(source_dir, 'processed.blend')
# bpy.ops.wm.save_as_mainfile(filepath=output_blend_file)

bpy.ops.wm.quit_blender()
