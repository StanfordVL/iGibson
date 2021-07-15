import os
import sys
import json
import math
import argparse
import matplotlib
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from collections import defaultdict 
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from scipy.spatial import ConvexHull, convex_hull_plot_2d
from shapely.geometry import Polygon as shape_poly

from utils.utils import *
from utils.semantics import *
from utils.scene_urdf import gen_scene_urdf,gen_orig_urdf,gen_orig_urdf_with_cabinet

import igibson

parser = argparse.ArgumentParser("Convert 3D-Front...")

parser.add_argument('--model_path', dest='model_path')
parser.add_argument('--save_root', dest='save_root', 
        default=os.path.join(igibson.threedfront_dataset_path, 'scenes'))

# https://stackoverflow.com/questions/13542855/
# algorithm-to-find-the-minimum-area-rectangle-for-given-points-in-order-to-comput/
# 33619018#33619018
def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """
    from scipy.ndimage.interpolation import rotate
    pi2 = np.pi/2.

    # get the convex hull for the points
    try:
        hull_points = points[ConvexHull(points).vertices]
    except:
        hull_points = points
    # calculate edge angles
    edges = np.zeros((len(hull_points)-1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([
        np.cos(angles),
        np.cos(angles-pi2),
        np.cos(angles+pi2),
        np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def mesh_to_xy_convex_hull(xyz):
    points = np.array([(x,z) for x,y,z in xyz]).astype(float)
    try:
        hull = ConvexHull(points)
        return np.array([points[hull.vertices,0], points[hull.vertices,1]]).transpose()
    except:
        return points

def get_bbox(mesh):
    xyz = np.array(mesh['xyz']).reshape([-1,3]).astype(float)
    return xyz.min(axis=0), xyz.max(axis=0)
            
def get_z_max(meshes):
    maxes = []
    for f in meshes:
        maxes.append(get_bbox(f)[1])
    max_all = np.vstack(maxes).max(axis=0)
    return max_all[1]

def get_scene_bbox(meshes):
    maxes = []
    mines = []
    for f in meshes:
        mmin, mmax = get_bbox(f)
        mines.append(mmin)
        maxes.append(mmax)
    min_all = np.vstack(mines).min(axis=0)
    max_all = np.vstack(maxes).max(axis=0)
    return min_all, max_all

def process_scene_wall(model_id):
    wall_cats = ['WallTop', 'Flue', 'Column']
    with open(model_id, 'r') as fp:
        data = json.load(fp)
    z_max = get_z_max(data['mesh'])
    wall_components = [f for f in data['mesh'] if f['type'] in wall_cats]
    polys = [mesh_to_xy_convex_hull(np.array(w['xyz']).reshape([-1,3])) for w in wall_components]
    rects = [minimum_bounding_rectangle(p) for p in polys]
    bboxes = [polygon_to_bbox(p[:,0], p[:,1], (0.,z_max), scale_factor=1.) for p in rects]
    return polys,bboxes

def intersects(poly, poly_list):
    for p in poly_list:
        if p.intersects(poly):
            inter = p.intersection(poly)
            if inter.area / p.area > 0.05:
                return True
    return False

def should_skip(model_id):
    with open(model_id, 'r') as fp:
        data = json.load(fp)
    for f in data['mesh']:
        if f['type'] in front_to_skip:
            return True
    return False

def process_hole_elements(model_id, cat):
    hole_cats = ['Door', 'Window', 'Hole']
    if cat not in hole_cats:
        raise ValueError('{} is not among the hole categories')
    with open(model_id, 'r') as fp:
        data = json.load(fp)
    components = [f for f in data['mesh'] if f['type'] == cat and get_bbox(f)[1][1] > 0.05]
    polys = [mesh_to_xy_convex_hull(np.array(w['xyz']).reshape([-1,3])) for w in components]
    zs = []
    for w in components:
        mmin, mmax = get_bbox(w) 
        zs.append((mmin[1], mmax[1]))
    filtered_poly = []
    filtered_indexes = []
    for i,p in enumerate(polys):
        ith_poly = shape_poly(p)
        if intersects(ith_poly, filtered_poly):
            continue
        filtered_poly.append(ith_poly)
        filtered_indexes.append(i)
    zs = [p for i,p in enumerate(zs) if i in filtered_indexes]
    polys = [p for i,p in enumerate(polys) if i in filtered_indexes]

    rects = [minimum_bounding_rectangle(p) for p in polys]
    bboxes = [polygon_to_bbox(p[:,0], p[:,1], z, scale_factor=1.) for p,z in zip(rects, zs) if z[0] != z[1]]
    return polys,bboxes
    
    
def process_scene_rooms(model_id):
    with open(model_id, 'r') as fp:
        data = json.load(fp)
    platforms = [f for f in data['mesh'] if f['type'] == 'CustomizedPlatform']
    platform_polys = [mesh_to_xy_convex_hull(np.array(w['xyz']).reshape([-1,3])) for w in platforms]
    to_fill = []
    if len(platforms) == 0:
        return to_fill
    # for each floor, get convex hull & draw on a picture
    components = [f for f in data['mesh'] if f['type'] == 'Floor']
    floor_polys = [mesh_to_xy_convex_hull(np.array(w['xyz']).reshape([-1,3])) for w in components]
    smin, smax = get_scene_bbox(data['mesh'])
    xmin,_,ymin = smin
    xmax,_,ymax = smax
    height = int(100. * (xmax - xmin)) + 20
    width = int(100. * (ymax - ymin)) + 20
    floor_img = Image.new('1', (height, width), 0)
    d = ImageDraw.Draw(floor_img)
    for coords in floor_polys:
        x = (coords[:,0] - xmin) * 100.
        y = (coords[:,1] - ymin) * 100.
        d.polygon(list(zip(x,y)),fill=1)   
    floor_mask = np.array(floor_img).astype(np.uint8)
    # then for each "CustomizedPlatform", check if floor is empty below
    # if so, add floor visual mesh
    platforms = [f for f in data['mesh'] if f['type'] == 'CustomizedPlatform']
    platform_polys = [mesh_to_xy_convex_hull(np.array(w['xyz']).reshape([-1,3])) for w in platforms]
    to_fill = []
    for coords in platform_polys:
        platform_img = Image.new('1', (height, width), 0)
        p = ImageDraw.Draw(platform_img)
        x = (coords[:,0] - xmin) * 100.
        y = (coords[:,1] - ymin) * 100.
        p.polygon(list(zip(x,y)),fill=1)
        platform_mask = np.array(platform_img).astype(np.uint8)
        prod = platform_mask * floor_mask
        if prod.sum() / platform_mask.sum() < 0.9:
            to_fill.append(coords)
    return to_fill

# things to process:
# WallTop, Door, Window, Hole, Floor
# wall also includes: flue, column
# to skip visual: CustomizedPlatform
# NOTE: check if customizedplatform is part of floor


def concatenate_meshes(meshes, save_path):
    num_verts = 0
    verts = []
    faces = []
    normals = []
    uvs = []
    for m in meshes:
        verts.append(np.array(m['xyz']).reshape([-1,3]))
        faces.append(np.array(m['faces']).reshape([-1,3]) + num_verts)
        normals.append(np.array(m['normal']).reshape([-1,3]))
        uvs.append(np.array(m['uv']).reshape([-1,2]))
        num_verts += verts[-1].shape[0]
    verts = np.vstack(verts)
    faces = np.vstack(faces)
    normals = np.vstack(normals)
    uvs = np.vstack(uvs)
    write_3dfront_obj(verts, faces, normals, uvs, save_path)
    
def write_3dfront_obj(xyz, faces, normals, uvs, savepath):
    # , vert, face, vtex, ftcoor, imgpath=None
    with open(savepath,'w') as fp:
        for v in xyz:
            fp.write('v {} {} {}\n'.format(v[0],v[2],v[1]))
        for vt in uvs:
            fp.write('vt {} {}\n'.format(vt[0],vt[1]))
#         for vn in normals:
#             fp.write('vn {} {} {}\n'.format(vn[0],vn[1], vn[2]))
        faces = faces + 1
        for f in faces:
            fp.write('f {a}/{a} {b}/{b} {c}/{c}\n'.format(a=f[2],b=f[1],c=f[0]))
            
def gen_static_cabinet_info(save_dir, save_name):
    urdf_dir = os.path.join(save_dir, 'urdf')
    os.makedirs(urdf_dir, exist_ok=True)
    model_name = os.path.basename(os.path.normpath(save_dir))
    vm_path = os.path.join(save_dir, 'shape', 'visual', 
                           '{}_vm.obj'.format(save_name))
    cm_path = os.path.join(save_dir, 'shape', 'collision', 
                           '{}_cm.obj'.format(save_name))
    cmd = '../../blender_utils/vhacd --input {} --output {}'.format(
                           vm_path, cm_path)
    subprocess.call(cmd,shell=True,
                    stdout=subprocess.DEVNULL)
    if not os.path.isfile:
        # vhacd failed. the cabinet is likely corrupted
        # using vm as cm
        cmd = 'cp {} {}'.format(vm_path, cm_path)
        subprocess.call(cmd,shell=True,
                    stdout=subprocess.DEVNULL)
    with open(os.path.join(urdf_dir, 
                '{}_{}s.urdf'.format(model_name,save_name)), 'w') as fp:
        fp.write(gen_scene_urdf(save_dir, model_name, save_name))
    with open(os.path.join(urdf_dir, 
                '{}_orig.urdf'.format(model_name)), 'w') as fp:
        fp.write(gen_orig_urdf_with_cabinet(model_name,save_name))



def export_visu_mesh(model_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(model_id, 'r') as fp:
        data = json.load(fp)
    walls = []
    ceilings = []
    floors = []
    cabinets = []
    for i,m in enumerate(data['mesh']):
        #mesh_to_obj(m, i, save_dir)
        category = m['type']
        if category in wall_cats:
            walls.append(m)
        elif category in ceiling_cats:
            ceilings.append(m)
        elif category in floor_cats:
            floors.append(m)
        elif category == 'Cabinet':
            cabinets.append(m)
        elif category == 'Door' or category == 'Hole':
            z_max = get_bbox(m)[1][1]
            if z_max < 0.05:
                floors.append(m)
    obj_dir = os.path.join(save_dir, 'shape', 'visual')
    os.makedirs(obj_dir, exist_ok=True)
    col_obj_dir = os.path.join(save_dir, 'shape', 'collision')
    os.makedirs(col_obj_dir, exist_ok=True)
    concatenate_meshes(walls, os.path.join(obj_dir, 'wall_vm.obj'))
    concatenate_meshes(ceilings, os.path.join(obj_dir, 'ceiling_vm.obj'))
    if len(cabinets) > 0:
        save_name = 'static_cabinet'
        vm_path = os.path.join(obj_dir, '{}_vm.obj'.format(save_name))
        concatenate_meshes(cabinets, vm_path)
        gen_static_cabinet_info(save_dir, save_name)
    for i,m in enumerate(floors):
        out_name = 'floor_{}_vm.obj'.format(i)
        out_path = os.path.join(obj_dir, out_name)
        xyz = np.array(m['xyz']).reshape([-1,3])
        faces = np.array(m['faces']).reshape([-1,3])
        normals = np.array(m['normal']).reshape([-1,3])
        uvs = np.array(m['uv']).reshape([-1,2])
        write_3dfront_obj(xyz, faces, normals, uvs, out_path)
        
def get_all_objs(model_id):
    with open (os.path.join(os.path.dirname(os.path.realpath(__file__))
               , 'utils', 'data', 'model_ig_cat.json')
               , 'r') as fp:
        model_to_cat = json.load(fp)
    with open (os.path.join(os.path.dirname(os.path.realpath(__file__))
               , 'utils', 'data', 'model_bbox.json')
               , 'r') as fp:
        model_to_bbox = json.load(fp)
    with open(model_id, 'r') as fp:
        data = json.load(fp)
        z_max = get_z_max(data['mesh'])
        model_jid = []
        model_uid = []
        model_bbox= []
        for ff in data['furniture']:
            if 'valid' in ff and ff['valid']:
                model_uid.append(ff['uid'])
                model_jid.append(ff['jid'])
                model_bbox.append(ff['bbox'])
        scene = data['scene']
        room = scene['room']

    object_used = defaultdict(lambda : -1)
    free_objs = []
    for r in room:
        room_id = r['instanceid']
        children = r['children']
        for c in children:
            ref = c['ref']
            if ref not in model_uid:
                continue
            obj_id = model_jid[model_uid.index(ref)]
            if obj_id not in model_to_bbox:
                continue
            pos = c['pos']
            pos_xy = np.array([pos[0], pos[-1]])

            rot = c['rot']
            ref = [0,0,1]
            axis = np.cross(ref, rot[1:])
            theta = np.arccos(np.dot(ref, rot[1:])) 
            og_cat = model_to_cat[obj_id]
            og_bbox = model_to_bbox[obj_id]
            if obj_id not in object_used:
                object_used[obj_id] = len(object_used) + 1
            instance_id = object_used[obj_id]

            scale = c['scale']
            scaled_bbox = np.array(og_bbox) * scale
            scaled_obj_frame = np.mean(scaled_bbox, axis=0)
            len_x,_,len_y = scaled_bbox.ptp(axis=0)
            edge_x_vanilla = np.array([len_x, 0])
            edge_y_vanilla = np.array([0, len_y])
            center_x, center_z, center_y = pos + scaled_obj_frame
            if np.sum(axis) != 0 and not math.isnan(theta):
                theta = theta
                axis = np.asarray(axis)
                axis = axis / math.sqrt(np.dot(axis, axis))
                a = math.cos(theta )
                b, c, d = -axis * math.sin(theta)
                aa, bb, cc, dd = a * a, b * b, c * c, d * d
                bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
                R = np.array([[aa + bb - cc - dd, 2 * (bd - ac)],
                                 [2 * (bd + ac), aa + dd - bb - cc]])
                edge_x = np.matmul(R, edge_x_vanilla.transpose())
                edge_y = np.matmul(R, edge_y_vanilla.transpose())
            else:
                edge_x = edge_x_vanilla
                edge_y = edge_y_vanilla
            normal = edge_y / np.linalg.norm(edge_y)
            z = (scaled_bbox+pos)[:,1]
            z[z < 0] = 0
            z[z > z_max] = z_max-0.01
            center = np.array([center_x, center_y])
            y = edge_x / 2.
            x = edge_y / 2.
            raw_pts = np.array([center - x - y,
                            center + x - y,
                            center + x + y,
                            center - x + y])
            if og_cat == 'bottom_cabinet' and z[0] > 0.1:
                og_cat = 'top_cabinet'
            free_objs.append(('{}={}'.format(og_cat, instance_id), 
                            {'edge_x':edge_y.tolist(), 
                             'edge_y':edge_x.tolist(), 
                             'center':center.tolist(), 
                             'normal':normal.tolist(),
                             'z':z.tolist(), 
                             'raw_pts':raw_pts.tolist()}))
    return free_objs

def get_scene_range(model_id):
    coords = [bbox.get_coords() for bbox in process_scene_wall(model_id)[1]]
    stacked = np.vstack(coords)
    xmin, ymin = stacked.min(axis=0)
    xmax, ymax = stacked.max(axis=0)
    max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
    max_length = np.ceil(max_length).astype(np.int)
    return max_length

def gen_room_maps(model_id, viz=False):
    max_length = get_scene_range(model_id)
    ins_image = Image.new('L', (2 * max_length * 100, 2 * max_length * 100), 0)
    d1 = ImageDraw.Draw(ins_image)
    sem_image = Image.new('L', (2 * max_length * 100, 2 * max_length * 100), 0)
    d2 = ImageDraw.Draw(sem_image)

    with open(model_id, 'r') as fp:
        data = json.load(fp)
        mesh_uid = []
        mesh_xyz = []
        for m in data['mesh']:
            category = m['type']
            #if category in wall_cats:
            mesh_uid.append(m['uid'])
            mesh_xyz.append(np.array(m['xyz']).reshape([-1,3]).astype(float))
        scene = data['scene']
        room = scene['room']
    for i, r in enumerate(room):
        room_id = r['instanceid']
        children = r['children']
        room_walls = []
        for c in children:
            ref = c['ref']
            if ref in mesh_uid:
                idx = mesh_uid.index(ref)
                v = mesh_xyz[idx]
                room_walls.append(v)
        verts = np.vstack(room_walls)
        hull_pts = (mesh_to_xy_convex_hull(verts) + max_length) * 100.
        room_id = rooms.index(front_all_rooms[r['type']])
        d1.polygon(list(zip(hull_pts[:,0],hull_pts[:,1])),fill=i+1)
        d2.polygon(list(zip(hull_pts[:,0],hull_pts[:,1])),fill=room_id+1)

    padded_image = Image.new('L', (3000, 3000), 0)
    og_size = sem_image.size
    padded_image.paste(sem_image, 
                        ((3000-og_size[0])//2,
                         (3000-og_size[1])//2))
    light_image = semmap_to_lightmap(np.array(padded_image))

    if viz:
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig,ax= plt.subplots(nrows=1,ncols=2,figsize=(13,5))
        divider = make_axes_locatable(ax[0])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[0].imshow(ins_image)
        fig.colorbar(im, cax=cax, orientation='vertical')

        divider = make_axes_locatable(ax[1])
        cax = divider.append_axes('right', size='5%', pad=0.05)
        im = ax[1].imshow(sem_image)
        fig.colorbar(im, cax=cax, orientation='vertical')

        plt.show()

    return ins_image, sem_image, light_image

def get_bbox_vol(bbox):
    coord_xy = shape_poly(bbox[1].get_coords())
    z = bbox[1].z
    return coord_xy.area * (z[1] - z[0])
    
def get_significant_overlaps(obj_bboxes, threshold=0.8, verbose=False):
    sig_overlap = set()
    for i in range(len(obj_bboxes)):
        for j in range(i + 1, len(obj_bboxes)):
            if i in sig_overlap:
                break
            if j in sig_overlap:
                continue
            bbox_i = obj_bboxes[i]
            bbox_j = obj_bboxes[j]
            intersection_vol = overlaps(bbox_i, bbox_j)
            if intersection_vol > 0:
                volume_i = get_bbox_vol(bbox_i)
                volume_j = get_bbox_vol(bbox_j)
                if volume_i < volume_j:
                    if (intersection_vol / volume_i) > threshold:
                        sig_overlap.add(i)
                        if verbose:
                            print(intersection_vol/ volume_i, bbox_i[0],  bbox_j[0])
                else:
                    if (intersection_vol / volume_j) > threshold:
                        sig_overlap.add(j)
                        if verbose:
                            print(intersection_vol/ volume_j, bbox_i[0],  bbox_j[0])
    return sig_overlap
            
def main():
    args = parser.parse_args()
    json_path = args.model_path
    if not os.path.isfile(json_path):
        raise ValueError('json file {} not found'.format(json_path))
    if should_skip(json_path):
        quit()
    model_id = os.path.splitext(
                os.path.basename(
                  os.path.normpath(json_path)))[0]
    save_dir = os.path.join(args.save_root, model_id)
    os.makedirs(save_dir, exist_ok=True)
    export_visu_mesh(json_path, save_dir)
    misc_dir = os.path.join(save_dir, 'misc')
    os.makedirs(misc_dir, exist_ok=True)
    with open(os.path.join(misc_dir, 'wall.json'), 'w') as fp:
        json.dump([bbox.as_dict() for bbox in process_scene_wall(json_path)[1]], fp)
    with open(os.path.join(misc_dir, 'window.json'), 'w') as fp:
        json.dump([bbox.as_dict() for bbox in process_hole_elements(json_path, 'Window')[1]], fp)
    with open(os.path.join(misc_dir, 'door.json'), 'w') as fp:
        json.dump([bbox.as_dict() for bbox in process_hole_elements(json_path, 'Door')[1]], fp)
    with open(os.path.join(misc_dir, 'hole.json'), 'w') as fp:
        json.dump([bbox.as_dict() for bbox in process_hole_elements(json_path, 'Hole')[1]], fp)
    with open(os.path.join(misc_dir, 'floor.json'), 'w') as fp:
        json.dump([("undefined",poly.tolist()) for poly in process_scene_rooms(json_path)], fp)

    objs = get_all_objs(json_path)
    obj_bboxes = [(cat, BBox(None,None,None,o)) for cat, o in objs]
    to_delete = get_significant_overlaps(obj_bboxes)
    objs = [o for i,o in enumerate(objs) if i not in to_delete]
    obj_bboxes = [(cat, BBox(None,None,None,o)) for cat, o in objs]
    lap_dict = defaultdict(lambda : [])
    for i in range(len(obj_bboxes)):
        for j in range(i+1, len(obj_bboxes)):
            if overlaps(obj_bboxes[i], obj_bboxes[j]):
                lap_dict[(i,obj_bboxes[i][1])].append((j,obj_bboxes[j][1]))
                lap_dict[(j,obj_bboxes[j][1])].append((i,obj_bboxes[i][1]))
                
    for _ in range(len(lap_dict)):
        overlaps_list = list(lap_dict.items())
        overlaps_list.sort(key=lambda x:-get_volume(x[0][1]))
        overlaps_list.sort(key=lambda x:-len(x[1]))
        o = overlaps_list[0]
        # try shrinking
        shrink_success = False
        bbox = o[0][1]
        edge_x_og = bbox.edge_x[:]
        edge_y_og = bbox.edge_y[:]
        for scale_factor in range(20):
            scale = 1. - 0.01 * scale_factor
            # try scale edge_x
            bbox.edge_x = edge_x_og * scale
            overlap = False
            for i in o[1]:
                if has_overlap(bbox, i[1]):
                    overlap=True
                    break
            if not overlap:
                shrink_success = True
                break
            bbox.edge_x = edge_x_og
            # try scale edge_y
            bbox.edge_y = edge_y_og * scale
            overlap = False
            for i in o[1]:
                if has_overlap(bbox, i[1]):
                    overlap=True
                    break
            if not overlap:
                shrink_success = True
                break
            bbox.edge_y = edge_y_og
            # try scale both
            bbox.edge_y = edge_y_og * scale
            bbox.edge_x = edge_x_og * scale
            overlap = False
            for i in o[1]:
                if has_overlap(bbox, i[1]):
                    overlap=True
                    break
            if not overlap:
                shrink_success = True
                break
            
        obj_bboxes[o[0][0]] = (obj_bboxes[o[0][0]][0], bbox)
        # update graph
        for j in o[1]:
            lap_dict[j].remove(o[0])
        del lap_dict[o[0]]

    with open(os.path.join(misc_dir, 'free_furniture.json'), 'w') as fp:
        json.dump([(cat, o.as_dict()) for cat, o in obj_bboxes], fp)

    layout_dir = os.path.join(save_dir, 'layout')
    os.makedirs(layout_dir, exist_ok=True)
    ins_image,sem_image,light_image = gen_room_maps(json_path)
    ins_image.save(os.path.join(layout_dir, 'floor_insseg_0.png'))
    sem_image.save(os.path.join(layout_dir, 'floor_semseg_0.png'))
    light_image.save(os.path.join(layout_dir, 'floor_lighttype_0.png'))

if __name__ == '__main__':
    main()
