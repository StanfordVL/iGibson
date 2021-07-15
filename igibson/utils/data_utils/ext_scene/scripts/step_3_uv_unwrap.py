import bpy
import math
import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '../../blender_utils')
sys.path.append(utils_dir)

from utils import redirect_output
model_dir = sys.argv[-1]

model_id = os.path.basename(model_dir)

obj_dir = os.path.join(model_dir, 'shape/visual') 

if os.path.isdir(obj_dir):
    objs = [a for a in os.listdir(obj_dir) if os.path.splitext(a)[1] == '.obj'] 

    blend_path = os.path.join(model_dir, '{}.blend'.format(model_id))

    ####################################
    # Clearing light, camera, cube 
    ####################################

    for on in bpy.context.scene.objects.keys():
        obj = bpy.context.scene.objects[on]
        bpy.data.objects.remove(obj)

    def check_has_vt_unwrapped_already(obj):
        with open(os.path.join(obj_dir, o), 'r') as fp:
            for l in fp.readlines():
                if l.startswith('vt '):
                    return True
        return False

    ####################################
    # Importing Objs
    ####################################

    with redirect_output():
        data_up, data_forward = ('Z', '-X')
        for o in objs:
            obj_path =  os.path.join(obj_dir, o)
            if check_has_vt_unwrapped_already(obj_path):
                continue
            bpy.ops.import_scene.obj(
                                filepath = obj_path,
                                axis_up = data_up, 
                                axis_forward = data_forward,
                                use_edges = True,
                                use_smooth_groups = True, 
                                use_split_objects = False,
                                use_split_groups = False,
                                use_groups_as_vgroups = False,
                                use_image_search = False,
                                split_mode = 'OFF')

        print(bpy.context.scene.objects.keys())


        ####################################
        # Per Object UV Unwrap  (preferred)
        ####################################
        vl = bpy.context.view_layer
        bpy.ops.object.select_all(action='DESELECT')
        for on in bpy.context.scene.objects.keys():
            obj = bpy.context.scene.objects[on]
            new_uv = bpy.context.scene.objects[on].data.uv_layers.new(name='obj_uv')
            vl.objects.active = obj
            obj.select_set(True)
            bpy.ops.object.editmode_toggle()
            bpy.ops.mesh.select_all(action="SELECT")
            bpy.ops.uv.smart_project(angle_limit=66, island_margin = 0.02)
            bpy.ops.object.editmode_toggle()
            bpy.ops.object.select_all(action='DESELECT')


        bpy.context.tool_settings.mesh_select_mode = (False, False, True)

        for on in bpy.context.scene.objects.keys():
            obj = bpy.context.scene.objects[on]
            vl.objects.active = obj
            obj.select_set(True)
            save_path = os.path.join(obj_dir, "{}.obj".format(on))
            bpy.ops.export_scene.obj(filepath=save_path, use_selection=True, 
                                     axis_up=data_up, axis_forward=data_forward, 
                                     use_materials=False,
                                     use_triangles=True,
                                     path_mode="COPY")

            obj.select_set(False)

bpy.ops.wm.quit_blender()
