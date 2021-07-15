import cv2
import os
import sys
import math
import json
import shutil
import random
import argparse
import matplotlib
from xml.dom import minidom
import numpy as np

from matplotlib.patches import Polygon
from shapely.geometry import Polygon as shape_poly
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from shapely.geometry import Polygon as shape_poly
from collections import defaultdict
from PIL import Image, ImageDraw

from utils.utils import BBox,polygon_to_bbox,has_overlap,get_volume, semmap_to_lightmap
from utils.svg_utils import PolygonWall, get_polygon, get_icon, get_points, get_direction
from utils.semantics import *

import igibson


parser = argparse.ArgumentParser("Convert Cubicasa5k...")

parser.add_argument('--model_dir', dest='model_dir')
parser.add_argument('--viz', dest='viz', action='store_true')
parser.add_argument('--save_root', dest='save_root', 
        default=os.path.join(igibson.cubicasa_dataset_path, 'scenes'))


def get_z_dim(e):
    desc = [p for p in e.childNodes if p.nodeName=='desc']
    if len(desc) == 0:
        raise ValueError('Furniture has no z dim annotation, should skip')
    dim_infos = desc[0].firstChild.nodeValue.split(' ')
    if len(dim_infos) != 4:
        raise ValueError('Furniture should have 4 dimension info. {}'.format(dim_infos))
    dim_dict = {}
    for d in dim_infos:
        if ":" not in d or len(d.split(':')) != 2:
            raise ValueError('Furniture dim should have : as separator, and two elements. {}'.format(dim_infos))
        k,v = d.split(':')
        dim_dict[k] = float(v)
    if 'Height' not in dim_dict or 'Elevation' not in dim_dict:
        raise ValueError('Height/Elevantion key not present in {}'.format(dim_infos))
    return dim_dict

def get_transformation_matrix(ee):
    transform = ee.getAttribute("transform")
    strings = transform.split(',')
    a = float(strings[0][7:])
    b = float(strings[1])
    c = float(strings[2])
    d = float(strings[3])
    e = float(strings[-2])
    f = float(strings[-1][:-1])

    M = np.array([[a, c, e],
                  [b, d, f],
                  [0, 0, 1]])

    if ee.parentNode.getAttribute("class") == "FixedFurnitureSet":
        parent_transform = ee.parentNode.getAttribute("transform")
        strings = parent_transform.split(',')
        a_p = float(strings[0][7:])
        b_p = float(strings[1])
        c_p = float(strings[2])
        d_p = float(strings[3])
        e_p = float(strings[-2])
        f_p = float(strings[-1][:-1])
        M_p = np.array([[a_p, c_p, e_p],
                        [b_p, d_p, f_p],
                        [0, 0, 1]])
        M = np.matmul(M_p, M)
    return M

def get_rotation_angle(ee):
    M = get_transformation_matrix(ee)
    if abs(M[0,0]) > 1.:
        return 0 if M[0,0] > 0 else math.pi
    return np.arccos(M[0,0])

def get_category(e):
    if "FixedFurniture " not in e.getAttribute("class"):
        return None
    class_name = e.getAttribute("class").split("FixedFurniture ")[-1]
    if class_name.startswith('ElectricalAppliance'):
        toks = class_name.split('ElectricalAppliance')
        if len(toks) != 2:
            return None
        class_name = toks[1].strip()
        if class_name == "":
            return None
    return class_name

def get_polygon(e):
    pol = next(p for p in e.childNodes if p.nodeName == "polygon")
    points = pol.getAttribute("points").split(' ')
    points = points[:-1]

    X, Y = np.array([]), np.array([])
    for a in points:
        y, x = a.split(',')
        X = np.append(X, np.round(float(x))) 
        Y = np.append(Y, np.round(float(y)))
    return X/100., Y/100.

def get_wall(svg, shape):
    wall_bboxes = []
    height, width = shape
    wall_image = Image.new('1', (height, width), 0)
    d = ImageDraw.Draw(wall_image)
    for e in svg.getElementsByTagName('g'):
        try: 
            if e.getAttribute("id") == "Wall":
                wall = PolygonWall(e, 1, shape)
                bbox = polygon_to_bbox(wall.Y, wall.X, (0,2.4), None)
                wall_bboxes.append(bbox)
                coords = bbox.get_coords()*100.
                y = coords[:,0]
                x = coords[:,1]
                d.polygon(list(zip(y,x)),fill=1)
        except ValueError as k:
            if str(k) != 'small wall':
                raise k
            continue
    return wall_bboxes, wall_image

def get_window_door(svg):
    window_bboxes = []
    door_bboxes = []
    for e in svg.getElementsByTagName('g'):
        if e.getAttribute("id") == "Window":
            X, Y = get_points(e)
            bbox = polygon_to_bbox(Y, X, (0.8,2.1), None)
            window_bboxes.append(bbox)
        if e.getAttribute("id") == "Door":
            # How to reperesent empty door space
            X, Y = get_points(e)
            bbox = polygon_to_bbox(Y, X, (0,2.2), None)
            door_bboxes.append(bbox)
    return window_bboxes, door_bboxes


def get_furniture(svg, wall_image):
    furniture_bboxes = []
    for e in svg.getElementsByTagName('g'):
        if "FixedFurniture " in e.getAttribute("class"):
            category = get_category(e)
            if category is None or category in cubi_to_skip:
                continue
            rr, cc, X, Y = get_icon(e)
            if len(X) != 4:
                continue
            z_dim = get_z_dim(e)
            z_min = (z_dim['Elevation']) / 100.
            z_max = z_min + z_dim['Height'] / 100.
            bbox = polygon_to_bbox(Y, X, (z_min,z_max), get_rotation_angle(e),flip_image=wall_image)
            furniture_bboxes.append((cubi_cat_mapping[category], bbox))
    return furniture_bboxes

def get_floor(svg):
    floor_polygons = []
    for e in svg.getElementsByTagName('g'):
        if "Space " in e.getAttribute("class"):
            room_type_raw = e.getAttribute("class").split(" ")[1]
            if room_type_raw not in cubi_all_rooms:
                room_type_ig = 'undefined'
            else:
                room_type_ig = cubi_all_rooms[room_type_raw]
            X, Y = get_points(e)
            floor_polygons.append((room_type_ig,
                np.vstack([Y/100.,X/100.]).transpose()))
    return floor_polygons



def main():
    args = parser.parse_args()
    model_dir = os.path.normpath(args.model_dir)
    print(model_dir)
    # model_id = "_".join(model_dir.split('/')[-2:])
    model_id = os.path.basename(os.path.normpath(model_dir))
    svg_file = os.path.join(model_dir, 'model.svg')
    img_path = os.path.join(model_dir, 'F1_scaled.png')
    svg = minidom.parse(svg_file)  # parseString also exists
    fplan = cv2.imread(img_path)
    fplan = cv2.cvtColor(fplan, cv2.COLOR_BGR2RGB)  # correct color channels
    height, width, nchannel = fplan.shape
    shape = height, width

    wall_bboxes, wall_image = get_wall(svg, shape)
    window_bboxes, door_bboxes = get_window_door(svg)
    furniture_bboxes = get_furniture(svg, wall_image)
    floor_polygons = get_floor(svg)

    overlaps = defaultdict(lambda : [])
    for i in range(len(furniture_bboxes)):
        for j in range(i+1, len(furniture_bboxes)):
            if has_overlap(furniture_bboxes[i][1],
                           furniture_bboxes[j][1]):
                overlaps[(i,furniture_bboxes[i][1])].append((j,furniture_bboxes[j][1]))
                overlaps[(j,furniture_bboxes[j][1])].append((i,furniture_bboxes[i][1]))

    to_delete = []


    for _ in range(len(overlaps)):
        overlaps_list = list(overlaps.items())
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

        if shrink_success:
            furniture_bboxes[o[0][0]] = (furniture_bboxes[o[0][0]][0], bbox)
        else:
            # add to delete
            to_delete.append(o[0][0])
        # update graph
        for j in o[1]:
            overlaps[j].remove(o[0])
        del overlaps[o[0]]

    for i in sorted(to_delete, reverse=True):
        del furniture_bboxes[i]


    ##################################
    # Splitting into separate floors #
    ##################################
    total_image = Image.new('1', (height, width), 0)
    d = ImageDraw.Draw(total_image)

    for group in [wall_bboxes, door_bboxes, window_bboxes]:
        for bbox in group:
            coords = bbox.get_coords()*100.
            y = coords[:,0]
            x = coords[:,1]
            d.polygon(list(zip(y,x)),fill=1)
    for _,bbox in furniture_bboxes:
        coords = bbox.get_coords()*100.
        y = coords[:,0]
        x = coords[:,1]
        d.polygon(list(zip(y,x)),fill=1)
    for _,coords in floor_polygons:
        y = coords[:,0]*100.
        x = coords[:,1]*100.
        d.polygon(list(zip(y,x)),fill=1)
    int_image = np.array(total_image).astype(np.uint8)
    binary = cv2.threshold(int_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    ret, labels = cv2.connectedComponents(binary)
    masks = []
    for label in range(1,ret):
        mask = np.zeros_like(int_image)
        mask[labels == label] = 1
        masks.append(mask)
    # iterate through list of masks, each of which is a floor
    for floor_i,mask_i in enumerate(masks):
        wall_bboxes_i = []
        window_bboxes_i = []
        door_bboxes_i= []
        furniture_bboxes_i = []
        floor_polygons_i = []
        for bbox in wall_bboxes:
            y,x = (bbox.center*100.).astype(int)
            if y >= height:
                y = height - 1
            if x >= width:
                x = width- 1
            if mask_i[x,y] == 1:
                wall_bboxes_i.append(bbox)
        for bbox in door_bboxes:
            y,x = (bbox.center*100.).astype(int)
            if y >= height:
                y = height - 1
            if x >= width:
                x = width- 1
            if mask_i[x,y] == 1:
                door_bboxes_i.append(bbox)
        for bbox in window_bboxes:
            y,x = (bbox.center*100.).astype(int)
            if y >= height:
                y = height - 1
            if x >= width:
                x = width- 1
            if mask_i[x,y] == 1:
                window_bboxes_i.append(bbox)
        for c,bbox in furniture_bboxes:
            y,x = (bbox.center*100.).astype(int)
            if y >= height:
                y = height - 1
            if x >= width:
                x = width- 1
            if mask_i[x,y] == 1:
                furniture_bboxes_i.append((c,bbox))
        for poly in floor_polygons:
            y,x = (np.array(
                    shape_poly(poly[1]).representative_point())*100.
                    ).astype(int)
            if y >= height:
                y = height - 1
            if x >= width:
                x = width- 1
            if mask_i[x,y] == 1:
                floor_polygons_i.append(poly)
        if len(wall_bboxes_i) < 4 or len(floor_polygons_i) < 1:
            # This suggests that the mask doesn't represent a floor
            continue
        model_dir = os.path.join(args.save_root, 
                '{}_floor_{}'.format(model_id,floor_i))
        os.makedirs(model_dir, exist_ok=True)
        save_dir = os.path.join(model_dir, 'misc')
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'wall.json'), 'w') as fp:
            json.dump([bbox.as_dict() for bbox in wall_bboxes_i], fp)
        with open(os.path.join(save_dir, 'window.json'), 'w') as fp:
            json.dump([bbox.as_dict() for bbox in window_bboxes_i], fp)
        with open(os.path.join(save_dir, 'door.json'), 'w') as fp:
            json.dump([bbox.as_dict() for bbox in door_bboxes_i], fp)
        with open(os.path.join(save_dir, 'furniture.json'), 'w') as fp:
            json.dump([(cat,bbox.as_dict()) for 
                cat,bbox in furniture_bboxes_i], fp)
        with open(os.path.join(save_dir, 'floor.json'), 'w') as fp:
            json.dump([(cat,poly.tolist()) for cat,poly in 
                        floor_polygons_i], fp)
            
        layout_dir = os.path.join(model_dir, 'layout')
        os.makedirs(layout_dir, exist_ok=True)

        coords = [bbox.get_coords() for bbox in wall_bboxes_i]
        stacked = np.vstack(coords)
        xmin, ymin = stacked.min(axis=0)
        xmax, ymax = stacked.max(axis=0)
        max_length = np.max([np.abs(xmin), np.abs(ymin), np.abs(xmax), np.abs(ymax)])
        max_length = np.ceil(max_length).astype(np.int)
        ins_image = Image.new('L', (2 * max_length * 100, 2 * max_length * 100), 0)
        d1 = ImageDraw.Draw(ins_image)
        sem_image = Image.new('L', (2 * max_length * 100, 2 * max_length * 100), 0)
        d2 = ImageDraw.Draw(sem_image)
        for i, (cat, poly) in enumerate(floor_polygons_i):
            room_id = rooms.index(cat)
            pts = [((x + max_length)*100.,(y + max_length)*100.) 
                    for x,y in poly]
            d1.polygon(pts,fill=i+1)
            d2.polygon(pts,fill=room_id+1)

        ins_image.save(os.path.join(layout_dir, 'floor_insseg_0.png'))
        sem_image.save(os.path.join(layout_dir, 'floor_semseg_0.png'))

        padded_image = Image.new('L', (3000, 3000), 0)
        og_size = sem_image.size
        padded_image.paste(sem_image, 
                            ((3000-og_size[0])//2,
                             (3000-og_size[1])//2))
        light_image = semmap_to_lightmap(np.array(padded_image))
        light_image.save(os.path.join(layout_dir, 'floor_lighttype_0.png'))

        if args.viz:
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
            png_path = os.path.join(save_dir, 'room_sem_viz.png')
            fig.savefig(png_path)

            wall_patches = [Polygon(bbox.get_coords(), True, color='b') 
                            for bbox in wall_bboxes_i]
            window_patches = [Polygon(bbox.get_coords(), True, color='b') 
                            for bbox in window_bboxes_i]
            door_patches = [Polygon(bbox.get_coords(), True, color='b') 
                            for bbox in door_bboxes_i]
            furn_patches = [Polygon(bbox.get_coords(), True, color='b') 
                            for c, bbox in furniture_bboxes_i]
            floor_patches = [Polygon(poly, True, color='b') 
                            for cat,poly in floor_polygons_i]

            quivers = [np.array([*bbox.center, *bbox.nrm]) 
                        for c,bbox in furniture_bboxes_i]

            fig, ax = plt.subplots(figsize=(11,11))
            p0 = PatchCollection(floor_patches, cmap=matplotlib.cm.jet, alpha=0.3)
            colors = np.random.random(10)
            p0.set_array(colors)
            ax.add_collection(p0)
            p = PatchCollection(wall_patches, alpha=0.2)
            ax.add_collection(p)
            p1 = PatchCollection(window_patches, alpha=1)
            ax.add_collection(p1)
            p2 = PatchCollection(door_patches, alpha=1)
            p2.set_array(np.random.random(10))
            ax.add_collection(p2)
            p3 = PatchCollection(furn_patches, cmap=matplotlib.cm.jet, alpha=1)
            colors = 100*np.random.random(10)
            p3.set_array(colors)
            ax.add_collection(p3)
            ax.set_ylim([0,width/100.])
            ax.set_xlim([0,height/100.])
            if len(quivers) > 0:
                x,y,u,v = np.vstack(quivers).transpose()
                ax.quiver(x, y, u, v, scale=20)
            png_path = os.path.join(save_dir, 'viz.png')
            fig.savefig(png_path)

if __name__ == '__main__':
    main()
