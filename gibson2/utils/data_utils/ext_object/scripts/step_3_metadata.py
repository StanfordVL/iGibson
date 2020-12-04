import numpy as np
import random
import os
import argparse
import json
import glob
import gibson2

parser = argparse.ArgumentParser("Generate Mesh meta-data...")
parser.add_argument('--input_dir', dest='input_dir')
parser.add_argument('--material', dest='material', default='wood')

def load_obj(fn):
    fin = open(fn, 'r')
    lines = [line.rstrip() for line in fin]
    fin.close()

    vertices = []
    for line in lines:
        if line.startswith('v '):
            vertices.append(np.float32(line.split()[1:4]))
    if len(vertices) <= 1:
        return None
    v = np.vstack(vertices)
    return v

def get_min_max(input_dir):
    mins = []
    maxs = []
    mesh_dir = os.path.join(input_dir, 'shape', 'collision')
    objs = glob.glob('{}/*.obj'.format(mesh_dir))
    for o in objs:
        mverts = load_obj(o)
        if mverts is None:
            continue
        mins.append( mverts.min(axis=0) )
        maxs.append( mverts.max(axis=0) )

    if len(mins) == 1:
        min_v = mins[0]
        max_v  = maxs[0]
    else:
        min_v = np.vstack(mins).min(axis=0)
        max_v  = np.vstack(maxs).max(axis=0)
    return min_v.astype(float), max_v.astype(float)

def gen_bbox(input_dir):
    min_c,max_c = get_min_max(input_dir)
    save_dict = { 'base_link_offset': tuple((max_c + min_c) / 2.),
                  'bbox_size' : tuple(max_c - min_c)}
    save_path = os.path.join(input_dir, 'misc', 'metadata.json')
    with open(save_path, 'w') as fp:
        json.dump(save_dict, fp)

def gen_material(input_dir, material_string):
    materials = material_string.split(',')
    material_dir = os.path.join(gibson2.ig_dataset_path, 'materials')
    material_json_file = os.path.join(material_dir, 'materials.json')
    assert os.path.isfile(material_json_file), \
        'cannot find material files: {}'.format(material_json_file)
    with open(material_json_file) as f:
        all_materials = json.load(f)
    for m in materials:
        assert m in all_materials, \
            'unknown material class: {}'.format(m)
    material_entry = {"1": materials}
    mesh_to_material = {}
    mesh_dir = os.path.join(input_dir, 'shape', 'visual')
    meshes = [o for o in os.listdir(mesh_dir) 
                if os.path.splitext(o)[-1] == '.obj']
    for m in meshes:
        mesh_to_material[m] = 1
    save_path = os.path.join(input_dir, 'misc', 'material_groups.json')
    with open(save_path, 'w') as fp:
        json.dump([material_entry, mesh_to_material], fp)

args = parser.parse_args()
if os.path.isdir(args.input_dir):
    misc_dir = os.path.join(args.input_dir, 'misc')
    os.makedirs(misc_dir, exist_ok=True)
    gen_bbox(args.input_dir)
    gen_material(args.input_dir, args.material)
