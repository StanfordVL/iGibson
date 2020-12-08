import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from xml.dom import minidom
import itertools
from collections import defaultdict
import numpy as np
from shapely.geometry import Polygon as shape_poly
from shapely.geometry import LineString as shape_string
from shapely.ops import cascaded_union
import warnings
import open3d as o3d
import argparse
import pickle
import json
import subprocess

from utils.Wall_bbox import Wall_bbox
import utils.earcut as earcut
from utils.utils import *
from utils.scene_urdf import gen_scene_urdf,gen_orig_urdf

parser = argparse.ArgumentParser("Generate mesh based on labelImg annotation...")

parser.add_argument('--model_dir', dest='model_dir') # directory wi=th json files
parser.add_argument('--skirt_file', dest='skirt_file', default='utils/data/skirt.npy')
parser.add_argument('--skip_saving', dest='skip_save', action='store_true')
parser.add_argument('--overwrite', dest='overwrite', action='store_true')

structure_classes = ['wall', 'door', 'window', 'floor', 
                     'fireplace', 'hole', 'stair']
ceiling_height = 2.4
fixed_scale_object = {'picture': [0.02]}
no_snap = ['fireplace', 'counter', 'TV']
cube_shape = ['counter', 'chest', 'carpet']

def gen_structure(wall_bbox,hole_bbox,floor_bbox, model_dir, skirt_pts, save_obj=False, overwrite=False):
    # finding correspondence between wall and windows
    # print('Examining walls and holes...')
    walls_poly = []
    for bbox in wall_bbox:
        polygon = shape_poly(bbox.get_coords())
        walls_poly.append(polygon)
    holes_poly = []
    for bbox,_ in hole_bbox:
        polygon = shape_poly(bbox.get_coords())
        holes_poly.append(polygon)
        
    walls_to_holes = defaultdict(lambda : [])
    for hi, hole in enumerate(holes_poly):
        has_intersect = False
        distances = []
        overlaps = [hole.intersects(wall) for wall in walls_poly]
        overlaps_area = [hole.intersection(wall).area  if overlap else 0 
                for overlap,wall in zip(overlaps, walls_poly) ]
        if sum(overlaps) > 1:
            max_idx = np.argmax(overlaps_area)
            for wi, o in enumerate(overlaps):
                if wi == max_idx:
                    walls_to_holes[wi].append(hole_bbox[hi])
                elif o:
                    snap_out_of_wall(hole_bbox[hi][0], wall_bbox[wi], 
                                     hole, walls_poly[wi])
        elif sum(overlaps) == 1:
            wi = overlaps.index(True)
            walls_to_holes[wi].append(hole_bbox[hi])
        else:
            continue

    if save_obj:
        structure_dir = os.path.join(model_dir, 'shape', 'visual')
        os.makedirs(structure_dir, exist_ok=True)
        vhacd_dir = os.path.join(model_dir, 'shape', 'collision')
        os.makedirs(vhacd_dir, exist_ok=True)

    verts = []
    faces = []

    structure_obj = []
    num_verts = 0
    for idx, w in enumerate(wall_bbox):
            
        wall = Wall_bbox(w, ceiling_height=w.z[1])
        if idx in walls_to_holes:
            for bbox,oname in walls_to_holes[idx]:
                new_bb = wall.add_hole(bbox)
                if oname != '':
                    structure_obj.append((new_bb, oname))
        v, f = wall.build()
        if save_obj:
            cm_path = os.path.join(vhacd_dir, 'wall_{}_cm.obj'.format(idx)) 
            if overwrite or not os.path.isfile(cm_path):
                wall.build_collision_mesh(cm_path)

        verts.append(v)
        f[:, :3] += num_verts
        faces.append(f)
        num_verts += v.shape[0]
        
        v, f = wall.add_skirt(skirt_pts)
        verts.append(v)
        f[:, :3] += num_verts
        faces.append(f)
        num_verts += v.shape[0]

    v = np.concatenate(verts, axis=0)
    f = np.concatenate(faces, axis=0)
    if save_obj:
        vm_path = os.path.join(structure_dir, 'wall_vm.obj'.format(idx))
        if overwrite or not os.path.isfile(vm_path):
            with open(vm_path, 'w') as fp:
                for vs in v.reshape([-1,3]):
                    fp.write('v {} {} {}\n'.format(*vs))
                for f1,f2,f3,fn in f:
                    fp.write('f {f1} {f2} {f3}\n'.format(f1=f1,f2=f2,f3=f3))

    ### Process floors
    floor_counter = 0
    if len(floor_bbox) > 0 and type(floor_bbox[0]) is not list:
        floor_ids = set([i for _,i in floor_bbox])
        if 0 in floor_ids:
            floor_id_remap = {}
            for i in floor_ids:
                floor_id_remap[i] = i+1
        else:
            floor_id_remap = {}
            for i in floor_ids:
                floor_id_remap[i] = i
            floor_id_remap[-1] = 0
        
        floor_polygons = defaultdict(lambda : [])
        for idx,(bbox,id) in enumerate(floor_bbox):
            if floor_id_remap[id] == 0:
                # print(floor_counter)
                if save_obj:
                    floor_path = os.path.join(structure_dir, 
                                  'floor_{}_vm.obj'.format(floor_counter))
                    v, f = gen_cube_obj(bbox, floor_path, 
                                        is_color=False, 
                                        should_save=(overwrite or not os.path.isfile(floor_path)))
                    v = np.asarray(v)
                    f = np.asarray(f)
                    verts.append(v)
                    f[:, :3] += num_verts
                    faces.append(f)
                    num_verts += v.shape[0]
                floor_counter += 1
            else:
                floor_polygons[floor_id_remap[id]].append(shape_poly(bbox.get_coords()))
    else:
        floor_polygons = {}
        for i,f in enumerate(floor_bbox):
            floor_polygons[i] = [shape_poly(f)]

    thickness = 0.2
    for k,polys in floor_polygons.items():
        if len(polys) > 1:
            x,y = cascaded_union(polys).exterior.coords.xy
        else:
            x,y = polys[0].exterior.coords.xy
        v = np.asarray([x,y]).T
        v = v[:-1]
        v_flat = np.reshape(v, -1)
        f_t = np.array(earcut.earcut(v_flat)).reshape((-1,3)) + 1
        v_z = np.asarray([0.,] * int(v.shape[0]))[:,np.newaxis]
        top_v = np.concatenate([v, v_z], axis=-1)
        back_v = top_v+0.0
        back_v[:,-1] = -thickness
        f_b = f_t + len(top_v)
        temp = f_b[:,1] + 0.0
        f_b[:,1] = f_b[:,0]
        f_b[:,0] = temp
        cross_faces = []
        curr_point = 1
        num_front_plane = len(top_v)
        for idx in range(num_front_plane):
            if idx == num_front_plane -1:
                next_point = 1
            else:
                next_point = curr_point + 1
                cross_faces.append((curr_point, next_point, curr_point + num_front_plane))
                cross_faces.append((curr_point + num_front_plane, next_point,
                                    next_point + num_front_plane))
            curr_point = next_point 
        v = np.concatenate([top_v, back_v])
        f = np.concatenate([f_t, f_b, cross_faces], axis=0)
        f_n  = np.asarray([0,] * int(f.shape[0]))[:,np.newaxis]
        f = np.concatenate([f, f_n], axis=-1)
        if save_obj:
            floor_path = os.path.join(structure_dir, 
                              'floor_{}_vm.obj'.format(floor_counter))
            if overwrite or not os.path.isfile(floor_path):
                with open(floor_path , 'w') as fp:
                    for vs in v.reshape([-1,3]):
                        fp.write('v {} {} {}\n'.format(*vs))
                    for f1,f2,f3,fn in f.reshape([-1,4]):
                        fp.write('f {f1} {f2} {f3}\n'.format(f1=f1,f2=f2,f3=f3))

        verts.append(v)
        f[:, :3] += num_verts
        faces.append(f)
        num_verts += v.shape[0]
        floor_counter += 1
        
    verts = np.concatenate(verts, axis=0)
    faces = np.concatenate(faces, axis=0)
    if save_obj:
        with open(os.path.join( model_dir, 'misc', 'structure.obj' ), 'w') as fp:
            for vs in verts.reshape([-1,3]):
                fp.write('v {} {} {}\n'.format(*vs))

            for f1,f2,f3,fn in faces.reshape([-1,4]):
                fp.write('f {f1} {f2} {f3}\n'.format(f1=f1,f2=f2,f3=f3))

    xa,ya,_ = verts.min(axis=0)
    x,y,_ = verts.max(axis=0)
    floor_coords = np.asarray([(x,y),(xa,y),(xa,ya),(x,ya)])
    total_floor = BBox(floor_coords,0, 0-thickness)

    if save_obj:
        floor_cm_path = os.path.join(vhacd_dir, 'floor_cm.obj') 
        if overwrite or not os.path.isfile(floor_cm_path):
            gen_cube_obj(total_floor, floor_cm_path, is_color=False)
        total_ceiling = BBox(floor_coords, ceiling_height + thickness, ceiling_height)
        ceiling_cm_path = os.path.join(vhacd_dir, 'ceiling_cm.obj') 
        if overwrite or not os.path.isfile(ceiling_cm_path):
            gen_cube_obj(total_ceiling, ceiling_cm_path, is_color=False)
        ceiling_vm_path = os.path.join(structure_dir, 'ceiling_vm.obj') 
        if overwrite or not os.path.isfile(ceiling_vm_path):
            gen_cube_obj(total_ceiling, ceiling_vm_path, is_color=False)
    
    bbox_dir = os.path.join(model_dir, 'misc',  'bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    all_objects = []
    for idx, (bbox, oclass) in enumerate(structure_obj):
        bbox_path = os.path.join(bbox_dir, '{}_{}_bbox.obj'.format(oclass, idx))
        x,y,z = bbox.get_scale()
        if x > y:
            continue
        gen_cube_obj(bbox, bbox_path, is_color=True)
        bbox_dict=bbox.as_dict()
        bbox_dict['category'] = oclass
        bbox_dict['instance'] = None 
        bbox_dict['is_fixed'] = True 
        bbox_dict['theta'] = bbox.get_rotation_angle() 
        all_objects.append(bbox_dict)
    return all_objects

def export_scene(fixed_obj_bbox,free_obj_bbox,wall_bbox,model_dir):
    walls_poly = []
    for bbox  in wall_bbox:
        polygon = shape_poly(bbox.get_coords())
        walls_poly.append(polygon)

    free_poly = []
    for _, bbox in free_obj_bbox:
        polygon = shape_poly(bbox.get_coords())
        free_poly.append(polygon)
    for i, obj in enumerate(free_poly):
        for wi, wall in enumerate(walls_poly):
            if obj.intersects(wall):
                snap_out_of_wall(free_obj_bbox[i][1], wall_bbox[wi], obj, wall)
    free_poly_snapped= [shape_poly(bbox.get_coords()) for _, bbox in free_obj_bbox]
    to_delete = []
    for i, obj in enumerate(free_poly_snapped):
        for wi, wall in enumerate(walls_poly):
            if obj.intersects(wall):
                to_delete.append(i)
    free_obj_bbox = [f for i,f in enumerate(free_obj_bbox) if i not in to_delete]

    fixed_poly = []
    for _, bbox in fixed_obj_bbox:
        polygon = shape_poly(bbox.get_coords())
        fixed_poly.append(polygon)
    vs = []
    fs = []
    all_objects = [] 

    bbox_dir = os.path.join(model_dir,'misc', 'bbox')
    os.makedirs(bbox_dir, exist_ok=True)
    for i,(oclass, bbox) in enumerate(free_obj_bbox):
        if oclass == 'shower_head':
            continue
        bbox_dict=bbox.as_dict()
        instance = None 
        if '=' in oclass:
            oclass,instance = oclass.split('=')
            instance = int(instance)
        bbox_path = os.path.join(bbox_dir, 'free_{}_{}_bbox.obj'.format(oclass, i))
        gen_cube_obj(bbox, bbox_path, is_color=True)
        bbox_dict['category'] = oclass
        bbox_dict['instance'] = instance 
        bbox_dict['is_fixed'] = False 
        bbox_dict['theta'] = bbox.get_rotation_angle() 
        all_objects.append(bbox_dict)

    for i,(oclass, bbox) in enumerate(fixed_obj_bbox):
        if oclass == 'shower_head':
            continue
        bbox_dict=bbox.as_dict()
        instance = None 
        if '=' in oclass:
            oclass,instance = oclass.split('=')
            instance = int(instance)
        bbox_path = os.path.join(bbox_dir, 'fixed_{}_{}_bbox.obj'.format(oclass, i))
        gen_cube_obj(bbox, bbox_path, is_color=True)
        bbox_dict['category'] = oclass
        bbox_dict['instance'] = instance 
        bbox_dict['is_fixed'] = True 
        bbox_dict['theta'] = bbox.get_rotation_angle() 
        all_objects.append(bbox_dict)
    
    return all_objects

def main():
    args = parser.parse_args()
    if not os.path.isdir(args.model_dir):
        raise ValueError('iGibson model directory does not exist: {}'
                .format(args.model_dir))
    model_dir = args.model_dir
    misc_path =os.path.join(model_dir, 'misc')
    os.makedirs(misc_path, exist_ok=True)
    skirt_pts = np.load(args.skirt_file)

    files = os.listdir(misc_path)
    global ceiling_height
    with open(os.path.join(misc_path, 'wall.json'), 'r') as fp:
        wall_bbox = [BBox(None,None,None,f) 
                          for f in json.load(fp)]
        ceiling_height = max([w.z[1] for w in wall_bbox])

    hole_bbox = []
    with open(os.path.join(misc_path, 'window.json'), 'r') as fp:
        hole_bbox.extend([(BBox(None,None,None,f), 'window') 
                          for f in json.load(fp)])
    with open(os.path.join(misc_path, 'door.json'), 'r') as fp:
        hole_bbox.extend([(BBox(None,None,None,f), 'door') 
                          for f in json.load(fp)])
    if 'hole.json' in files:
        with open(os.path.join(misc_path, 'hole.json'), 'r') as fp:
            hole_bbox.extend([(BBox(None,None,None,f), '') 
                          for f in json.load(fp)])
    if 'furniture.json' in files:
        with open(os.path.join(misc_path, 
                               'furniture.json'), 'r') as fp:
            fixed_obj_bbox = [(c,BBox(None,None,None,f)) 
                             for c,f in json.load(fp)]
    else:
        fixed_obj_bbox = []
    with open(os.path.join(misc_path, 'floor.json'), 'r') as fp:
        floor_bbox = [b for _,b in json.load(fp)]
    if 'free_furniture.json' in files:
        with open(os.path.join(misc_path, 
                               'free_furniture.json'), 'r') as fp:
            free_obj_bbox = [(c,BBox(None,None,None,f)) 
                             for c,f in json.load(fp)]
    else:
        free_obj_bbox = []

    structure_objs = gen_structure(wall_bbox, hole_bbox, floor_bbox, model_dir, 
                                   skirt_pts, save_obj=(not args.skip_save), overwrite=args.overwrite)
    all_objects = export_scene(fixed_obj_bbox,free_obj_bbox,wall_bbox,model_dir)
    
    all_objects.extend(structure_objs)
    json_path = os.path.join(misc_path, "all_objs.json")
    with open(json_path, 'w') as outfile:
        json.dump(all_objects, outfile)

    model_name = os.path.basename(model_dir)
    urdf_dir = os.path.join(model_dir, 'urdf')
    os.makedirs(urdf_dir,exist_ok=True)

    for component in ['wall', 'ceiling', 'floor']:
        with open(os.path.join(urdf_dir, 
                    '{}_{}s.urdf'.format(model_name,component)), 'w') as fp:
            fp.write(gen_scene_urdf(model_dir, model_name, component))
    with open(os.path.join(urdf_dir, 
                '{}_orig.urdf'.format(model_name)), 'w') as fp:
        fp.write(gen_orig_urdf(model_name))

    layout_dir = os.path.join(model_dir, 'layout')
    os.makedirs(layout_dir,exist_ok=True)
   
if __name__ == '__main__':
    main()

