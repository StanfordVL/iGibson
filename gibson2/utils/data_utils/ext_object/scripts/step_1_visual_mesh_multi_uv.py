import bpy
import math
import sys
import os
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '../../blender_utils')
sys.path.append(utils_dir)

from utils import import_obj_folder, export_ig_object, bake_model, clean_unused


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

axis = ['X', 'Y', 'Z', '-X', '-Y', '-Z']
import_axis_up = get_arg(sys.argv, '--up', default='Z')
if import_axis_up not in axis:
    raise ValueError(
        'Axis up not supported: {} (should be among X,Y,Z,-X,-Y,-Z)'
        .format(import_axis_up))

import_axis_forward = get_arg(sys.argv, '--forward', default='X')
if import_axis_forward not in axis:
    raise ValueError(
        'Axis forward not supported: {} (should be among X,Y,Z,-X,-Y,-Z)'
        .format(import_axis_forward))

source_blend_file = get_arg(sys.argv, '--source_blend_file')
if source_blend_file is None:
    raise ValueError('Source directory not specified.')
dest_dir = get_arg(sys.argv, '--dest_dir')
if dest_dir is None:
    raise ValueError('Destination directory not specified.')
os.makedirs(dest_dir, exist_ok=True)


# import_obj_folder(model_id, source_dir,
#                  up=import_axis_up,
#                  forward=import_axis_forward)

# open blend file here source_blend_file
bpy.ops.wm.open_mainfile(filepath=source_blend_file)

#############################################
# Importing obj files from source dir
#############################################

# uncomment this part for evermotion models
# for on in bpy.context.scene.objects.keys():
#     obj = bpy.context.scene.objects[on]
#     if not 'AM' in on or 'Cam' in on:
#         bpy.data.objects.remove(obj)
#         print(on)
#     # only keep evermotion models
# clean_unused()

#############################################
# Decimate the mesh
#############################################
max_num_verts = 10000

meshes = []
for obj in bpy.data.objects:
    if obj.type == 'MESH':
        meshes.append(obj)
total_num_verts = sum([len(mesh.data.vertices) for mesh in meshes])
print('total_num_verts (before decimation)', total_num_verts)
print('max_num_verts', max_num_verts)
if total_num_verts > max_num_verts:
    ratio = float(max_num_verts) / total_num_verts
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.editmode_toggle()
    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.decimate(ratio=ratio)
    bpy.ops.object.editmode_toggle()
    print('ratio', ratio)
    meshes = []
    for obj in bpy.data.objects:
        if obj.type == "MESH":
            meshes.append(obj)
    total_num_verts = sum([len(mesh.data.vertices) for mesh in meshes])
    print('total_num_verts (after decimation)', total_num_verts)

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
        bpy.ops.object.select_all(action='SELECT')
        bpy.ops.object.mode_set(mode='OBJECT')
        vl = bpy.context.view_layer
        bpy.ops.object.select_all(action='DESELECT')
        for on in bpy.context.scene.objects.keys():
            obj = bpy.context.scene.objects[on]
            new_uv = bpy.context.scene.objects[on].data.uv_layers.new(
                name='obj_uv')
            new_uv.active = True
            vl.objects.active = obj
            obj.select_set(True)
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66, island_margin=0.02)
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode='OBJECT')


#############################################
# Export models
#############################################
export_ig_object(dest_dir, save_material=not should_bake)

has_glass = False
# detect glass
for i in range(len(bpy.data.materials)):
    if 'glass' in bpy.data.materials[i].name.lower():
        has_glass = True

#############################################
# Optional Texture Baking
#############################################
if should_bake:
    mat_dir = os.path.join(dest_dir, 'material')
    os.makedirs(mat_dir, exist_ok=True)

    # bpy.ops.wm.open_mainfile(filepath=blend_path)
    # import_ig_object(model_root, import_mat=True)
    for obj in bpy.context.scene.objects:
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.join()

    channels = {
        'DIFFUSE': (2048, 32),
        'ROUGHNESS': (1024, 16),
        'METALLIC': (1024, 16),
        'NORMAL': (1024, 16),
    }
    if has_glass:
        channels['TRANSMISSION'] = (1024, 32)
        # add world light
        world = bpy.data.worlds['World']
        world.use_nodes = True

        # changing these values does affect the render.
        bg = world.node_tree.nodes['Background']
        bg.inputs[0].default_value[:3] = (1, 1, 1)
        bg.inputs[1].default_value = 1.0

    bake_model(mat_dir, channels, overwrite=True, add_uv_node=True)


bpy.ops.wm.quit_blender()

# optionally save blend file
# output_blend_file = os.path.dirname(
#     source_blend_file) + 'processed_' + os.path.basename(source_blend_file)
# bpy.ops.wm.save_as_mainfile(filepath=output_blend_file)
