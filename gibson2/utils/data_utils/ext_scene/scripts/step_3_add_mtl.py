import os
import glob
import random
import subprocess
import argparse
import json

use_mat = 'mtllib {}.mtl\nusemtl default\n'
wall_default='''newmtl default
Ns 225.000000
Ka 1.000000 1.000000 1.000000
Kd 0.800000 0.800000 0.800000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
map_Kd ../../../../materials/default_material/wall/diffuse.png
map_Pr ../../../../materials/default_material/wall/roughness.png
map_bump ../../../../materials/default_material/wall/normal.png
'''
floor_default='''newmtl default
Ns 225.000000
Ka 1.000000 1.000000 1.000000
Kd 0.800000 0.800000 0.800000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
map_Kd ../../../../materials/default_material/floor/diffuse.png
map_Pr ../../../../materials/default_material/floor/roughness.png
map_bump ../../../../materials/default_material/floor/normal.png
'''

parser = argparse.ArgumentParser("Add default mtl files...")
parser.add_argument('--input_dir', dest='input_dir')

def add_mat(input_mesh, output_mesh, mat_name):
    with open(input_mesh, 'r') as fin:
        lines = fin.readlines()
    for l in lines:
        if l == 'mtllib {}.mtl\n'.format(mat_name):
            return
    with open(output_mesh, 'w') as fout:
        fout.write(use_mat.format(mat_name))
        for line in lines:
            if not line.startswith('o') and not line.startswith('s'):
                fout.write(line)

def gen_scene_mat(scene_dir):
    misc_dir = os.path.join(scene_dir, 'misc')
    mesh_dir = os.path.join(scene_dir, 'shape', 'visual') 

    wall_mat = {"1": ["bricks", "concrete", "paint", "plaster"]}
    wall_mat_map = {}
    mat_name = 'wall'
    mtl_path = os.path.join(mesh_dir, '{}.mtl'.format(mat_name))
    with open(mtl_path, 'w') as fp:
        fp.write(wall_default)
    for f in os.listdir(mesh_dir):
        if ('wall' in f or 'skirt' in f) and os.path.splitext(f)[-1] == '.obj':
            wall_mat_map[f] = 1
            obj_path = os.path.join( mesh_dir, f )
            add_mat( obj_path, obj_path, mat_name )
    with open(os.path.join(misc_dir, 
              'walls_material_groups.json'), 'w') as fp:
        json.dump([wall_mat, wall_mat_map], fp)

    ceiling_mat = {"1": ["concrete", "paint", "plaster"]}
    ceiling_mat_map = {}
    mat_name = 'ceiling'
    mtl_path = os.path.join(mesh_dir, '{}.mtl'.format(mat_name))
    with open(mtl_path, 'w') as fp:
        fp.write(wall_default)
    for f in os.listdir(mesh_dir):
        if 'ceiling' in f and os.path.splitext(f)[-1] == '.obj':
            ceiling_mat_map[f] = 1
            obj_path = os.path.join( mesh_dir, f )
            add_mat( obj_path, obj_path, mat_name )
    with open(os.path.join(misc_dir, 
              'ceilings_material_groups.json'), 'w') as fp:
        json.dump([ceiling_mat, ceiling_mat_map], fp)

    floor_mat = {"1": ["asphalt", "concrete", "fabric_carpet", 
                       "marble", "planks", "terrazzo", 
                       "tiles", "wood_floor"]}
    floor_mat_map = {}
    for f in os.listdir(mesh_dir):
        if 'floor' not in f or os.path.splitext(f)[-1] != '.obj':
            continue
        floor_mat_map[f] = 1
        mat_name = '_'.join(f.split('_')[:2])
        mtl_path = os.path.join(mesh_dir, '{}.mtl'.format(mat_name))
        with open(mtl_path, 'w') as fp:
            fp.write(floor_default)
        obj_path = os.path.join( mesh_dir, f )
        add_mat( obj_path, obj_path, mat_name )
    with open(os.path.join(misc_dir, 
              'floors_material_groups.json'), 'w') as fp:
        json.dump([floor_mat, floor_mat_map], fp)
    

args = parser.parse_args()
if os.path.isdir(args.input_dir):
    gen_scene_mat(args.input_dir) 
