import bpy
import math
import sys
import os
import subprocess

script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '../../blender_utils')
sys.path.append(utils_dir)

from utils import import_obj_folder, export_ig_object, bake_model, clean_unused

#############################################
# Parse command line arguments
#############################################

def get_arg(argv, flag, default=None):
    if flag in argv:
        return argv[argv.index(flag) + 1]
    return default

should_bake = "--bake" in sys.argv

axis=['X','Y','Z','-X','-Y','-Z']
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

source_dir = get_arg(sys.argv, '--source_dir') 
if source_dir is None:
    raise ValueError('Source directory not specified.')
dest_dir = get_arg(sys.argv, '--dest_dir') 
if dest_dir is None:
    raise ValueError('Destination directory not specified.')
os.makedirs(dest_dir, exist_ok=True)

model_id = os.path.basename(source_dir)

#############################################
# Importing obj files from source dir
#############################################
for on in bpy.context.scene.objects.keys():
    obj = bpy.context.scene.objects[on]
    bpy.data.objects.remove(obj)
clean_unused()

import_obj_folder(model_id, source_dir, 
                  up=import_axis_up, 
                  forward=import_axis_forward)

#############################################
# Optional UV Unwrapping
# This only needed if baking will be performed
#############################################
if should_bake:
    uv_unwrapped = True 
    for o in bpy.context.scene.objects:
        if not o.data.uv_layers:
            uv_unwrapped = False
    if not uv_unwrapped:
        bpy.ops.object.mode_set(mode='OBJECT')
        vl = bpy.context.view_layer
        bpy.ops.object.select_all(action='DESELECT')
        for on in bpy.context.scene.objects.keys():
            obj = bpy.context.scene.objects[on]
            new_uv = bpy.context.scene.objects[on].data.uv_layers.new(
                        name='obj_uv')
            vl.objects.active = obj
            obj.select_set(True)
        bpy.ops.object.editmode_toggle()
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.uv.smart_project(angle_limit=66, island_margin = 0.02)
        bpy.context.tool_settings.mesh_select_mode = (False, False, True)
        bpy.ops.object.mode_set(mode='OBJECT')


#############################################
# Export models
#############################################
export_ig_object(dest_dir, save_material=not should_bake)


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
    'DIFFUSE':(2048, 32),
    'ROUGHNESS':(1024, 16),
    'METALLIC':(1024, 16),
    'NORMAL':(1024, 16),
    }
    bake_model(mat_dir, channels, overwrite=True)

bpy.ops.wm.quit_blender()


