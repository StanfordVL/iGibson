import bpy
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '../../blender_utils')
sys.path.append(utils_dir)
from collections import defaultdict
from utils import clean_unused, import_obj_folder

model_id =  sys.argv[-3]
obj_dir =  sys.argv[-2]
save_dir =  sys.argv[-1]
os.makedirs(save_dir, exist_ok=True)

for on in bpy.context.scene.objects.keys():
    obj = bpy.context.scene.objects[on]
    bpy.data.objects.remove(obj)
clean_unused()
import_obj_folder('object', obj_dir)
for obj in bpy.context.scene.objects:
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj 
bpy.ops.object.select_all(action='SELECT')
save_path = os.path.join(save_dir, "{}_cm.obj".format(model_id))
bpy.ops.export_scene.obj(filepath=save_path, use_selection=True, 
                         axis_up='Z', axis_forward='X', 
                         use_materials=False,
                         use_normals=False, use_uvs=False, 
                         use_triangles=True,
                         path_mode="COPY")
