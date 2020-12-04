import bpy
import glob
import os
from typing import Tuple
import sys
import math
import json
import random
import numpy as np
script_dir = os.path.dirname(os.path.abspath(__file__))
utils_dir = os.path.join(script_dir, '../../blender_utils')
sys.path.append(utils_dir)
from utils import import_obj_folder,export_obj_folder,clean_unused

obj_dir =  sys.argv[-2]
save_dir =  sys.argv[-1]
os.makedirs(save_dir, exist_ok=True)

for on in bpy.context.scene.objects.keys():
    obj = bpy.context.scene.objects[on]
    bpy.data.objects.remove(obj)
clean_unused()

import_obj_folder('object', obj_dir)
bpy.ops.object.select_all(action='DESELECT')

for obj in bpy.context.scene.objects:
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj 
    # bpy.ops.object.editmode_toggle()
    # bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.separate(type='LOOSE') 
    # bpy.ops.object.mode_set(mode='OBJECT')
    obj.select_set(False)
    bpy.ops.object.select_all(action='DESELECT')

export_obj_folder(save_dir, skip_empty=True)
bpy.ops.wm.quit_blender()



