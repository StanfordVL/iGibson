from xml.dom import minidom
from shapely.geometry import Polygon as shape_poly
from shapely.geometry import LineString as shape_string
import numpy as np
import random
import os
import cv2
from PIL import Image


def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang2 - ang1) % (2 * np.pi)

def pad_to(im, size = 3000):
    row, col = im.shape[:2]
    pad_r = (size - row) // 2
    pad_c = (size - col) // 2
    border = cv2.copyMakeBorder(
        im,
        top=pad_r,
        bottom=pad_r,
        left=pad_c,
        right=pad_c,
        borderType=cv2.BORDER_CONSTANT,
        value=[0]
    )
    return border
def semmap_to_lightmap(sem):
    def gkern(l=10, sig=5):
        """\
        creates gaussian kernel with side length l and a sigma of sig
        """

        ax = np.linspace(-(l - 1) / 2., (l - 1) / 2., l)
        xx, yy = np.meshgrid(ax, ax)

        kernel = np.exp(-0.5 * (np.square(xx) + np.square(yy)) / np.square(sig))

        return kernel / np.sum(kernel)
    kernel = gkern()
    sem_map = np.array(sem)
    kitchen_bathroom = np.logical_or(np.logical_or(sem_map == 10, sem_map == 1), sem_map==19)
    kitchen_bathroom_filtered = cv2.dilate(kitchen_bathroom.astype(np.float32),   
                                           kernel.astype(np.float32))
    kitchen_bathroom_filtered = cv2.filter2D(kitchen_bathroom_filtered.astype(np.float32), 
                                             -1,   kernel.astype(np.float32))
    return Image.fromarray((kitchen_bathroom_filtered * 255).astype(np.uint8))

class BBox(object):
    
    def __init__(self, raw_pts=None, zmin=None, zmax=None, from_dict=None):
        if raw_pts is None:
            self.load_dict(from_dict)
        else:
            self.raw_pts = raw_pts
            self.z = (zmin, zmax)
            self.init_box()

    def init_box(self):
        self.center = self.raw_pts.mean(axis=0)
        self.calc_nrm()

    def get_scale(self):
        return np.linalg.norm(self.edge_x), np.linalg.norm(self.edge_y), self.z[1]-self.z[0] 

    def calc_nrm(self):
        if self.raw_pts[0,0] > self.raw_pts[1,0]: id1, id2 = 1, 0
        else: id1, id2 = 0, 1

        direction = self.raw_pts[id1, :] - self.raw_pts[id2, :] 
        dist = np.linalg.norm(direction)
        des_len = 0.4
        nrm = np.array([-direction[1] * des_len / dist, direction[0] * des_len / dist]) 
        nrm_start = (self.raw_pts[id1] + self.raw_pts[id2]) / 2.
        cand1 = nrm_start + nrm
        cand2 = nrm_start - nrm
        if np.linalg.norm(cand1 - self.center) > np.linalg.norm(cand2 - self.center):
            flip_factor = 1
        else:
            flip_factor = -1
        nrm = nrm * flip_factor
        self.nrm=nrm / np.linalg.norm(nrm)
        # self.flip_factor = flip_factor
        # return nrm
        self.edge_y = direction * flip_factor
        self.edge_x = np.linalg.norm(self.raw_pts[0,:] - self.raw_pts[3,:]) * self.nrm 

    def get_coords(self):
        '''
        Return the vertices of the bounding box, in order of BL,BR,TR,TL
        '''
        x = self.edge_x / 2.
        y = self.edge_y / 2.
        return np.array([self.center - x - y,
                        self.center + x - y,
                        self.center + x + y,
                        self.center - x + y])

    def get_rotation_angle(self, CAD_dir=(0,1)):
        # dir_y, dir_x = self.nrm / np.linalg.norm(self.nrm)
        dir_y, dir_x = self.nrm[:]
        #theta = np.arcsin(np.cross([1,0], [dir_x, dir_y]))
        return angle_between(CAD_dir, (dir_x,dir_y))

    def get_center(self):
        return (*self.center, (self.z[0] + self.z[1])/2.)
    def get_nrm(self):
        return self.nrm

    def as_dict(self):
        return {
            'normal' : tuple(self.nrm),
            'center' : tuple(self.center),
            'edge_x' : tuple(self.edge_x),
            'edge_y' : tuple(self.edge_y),
            'raw_pts' : self.raw_pts.tolist(),
            'theta' : self.get_rotation_angle(),
            'z' : tuple(self.z)}

    def load_dict(self, d):
        self.nrm = np.array(d['normal'])
        self.center = np.array(d['center'])
        self.edge_x = np.array(d['edge_x'])
        self.edge_y = np.array(d['edge_y'])
        self.z = d['z']
        self.raw_pts = np.array(d['raw_pts'])


def polygon_to_bbox(Y,X,Z,rotation=None,flip_image=None, scale_factor=100.):
    pts = np.vstack((Y,X)).transpose()
    center = pts.mean(axis=0)
    dists = np.array([np.linalg.norm(pts[i] - pts[i-1]) for i in range(4)])
    max_idx = np.argmax(dists)
    p0,p1 = pts[max_idx],pts[max_idx-1]
    edge_y = (p1-p0)
    edge_y_dir = edge_y/np.linalg.norm(edge_y)
    edge_y_projected = np.array([np.dot(pts[i]-p1,edge_y_dir) for i in range(4)])
    edge_y_raw = edge_y_dir * np.ptp(edge_y_projected)
    mid_y =  (p0+p1)/2.
    edge_x_crook = (center - mid_y)
    edge_x_raw = 2*(edge_x_crook - (edge_y_dir * np.dot(edge_x_crook, edge_y_dir)))
    edge_x_dir = edge_x_raw / np.linalg.norm(edge_x_raw)
    # so we have:
    # edge_x, edge_y, center here
    # Z is given as input
    # we need to figure out normal direction (aka theta)
    # and reorient X/Y accordingly
    if rotation is None:
        # this means that the object is wall/door/window
        if np.linalg.norm(edge_x_raw) > np.linalg.norm(edge_y_raw):
            edge_x = edge_y_raw
            edge_y = edge_x_raw
            nrm = edge_y_dir
        else:
            edge_y = edge_y_raw
            edge_x = edge_x_raw
            nrm = edge_x_dir
    else:
        # this means that this is a fixed furniture
        forward_direction = np.array([1,0])
        rotated = np.matmul(np.asarray([[np.cos(rotation),-np.sin(rotation)],
                              [np.sin(rotation),np.cos(rotation)]]), forward_direction)
        fit_x = np.dot(edge_x_dir, rotated)
        fit_y = np.dot(edge_y_dir, rotated)
        
        if abs(fit_y) > abs(fit_x):
            edge_x = edge_y_raw
            edge_y = edge_x_raw
            nrm = edge_y_dir * np.sign(fit_y)
        else:
            edge_y = edge_y_raw
            edge_x = edge_x_raw 
            nrm = edge_x_dir * np.sign(fit_x)

    if flip_image is not None:
        if should_flip(center, nrm, flip_image):
            nrm = -nrm
    bbox_dict = {'center':center/scale_factor, 'z':Z, 'edge_x':edge_x/scale_factor, 
                 'edge_y':edge_y/scale_factor, 'normal':nrm, 'raw_pts':pts/scale_factor}
    return BBox(None,None,None,bbox_dict)

def should_flip(center, nrm, flip_image):
    flip_image_np = np.asarray(flip_image).astype(int)
    #plt.imshow(flip_image_np)
    # normal direction
    normal_dist = 0
    width,height = flip_image_np.shape
    for i in range(1000):
        y,x = np.round(center + i * nrm).astype(int)
        if x >= width or x < 0 or y >= height or y < 0:
            normal_dist = 2000
            break
        if flip_image_np[x,y]:
            normal_dist = i
            break
    # flipped direction
    flip_dist = 0
    for i in range(1000):
        y,x = np.round(center - i * nrm).astype(int)
        if x >= width or x < 0 or y >= height or y < 0:
            flip_dist = 2000
            break
        if flip_image_np[x,y]:
            flip_dist = i
            break        
            
    y,x = np.round(center).astype(int)
    #print(nrm, center, normal_dist, flip_dist)
    return flip_dist > normal_dist

# def squash_to_size(xmin,xmax,ymin,ymax,scale):
def squash_to_size(bbox,scale):
    size = random.choice(scale)
    print('squashing object...')
    x,y,_ = bbox.get_scale()
    # print(bbox.get_scale())
    if x < y:
        bbox.edge_x = bbox.edge_x / np.linalg.norm(bbox.edge_x) * size
    else:
        bbox.edge_y = bbox.edge_y / np.linalg.norm(bbox.edge_y) * size

def get_unique(doc, val):
    it = doc.getElementsByTagName(val)
    if len(it) > 1:
        raise ValueError('value not unique...')
    return it[0].firstChild.nodeValue


def snap_out_of_wall(obj, wall, obj_poly, wall_poly):
    intersection = wall_poly.intersection(obj_poly)
    if type(intersection) == shape_string:
        overlap_margin = 0.001
        for move in [(overlap_margin, 0), (-overlap_margin, 0), 
                     (0, overlap_margin), (0, -overlap_margin)]:
            temp_obj = shape_poly(obj.get_coords() + move)
            if not temp_obj.intersects(wall_poly):
                obj.center += move
                return
    elif type(intersection) == shape_poly:
        xy = np.array(intersection.exterior.coords.xy)
        ptp = xy.ptp(axis=1) + 0.001
        for move in [(ptp[0], 0), (-ptp[0], 0), 
                     (0, ptp[1]), (0, -ptp[1])]:
            temp_obj = shape_poly(obj.get_coords() + move)
            if not temp_obj.intersects(wall_poly):
                obj.center += move
                return

def snap_on_wall(obj, wall, obj_poly, wall_poly):
    if wall_poly.intersects(obj_poly):
        intersection = wall_poly.intersection(obj_poly)
        if type(intersection) == shape_string:
            return
        elif type(intersection) == shape_poly:
            xy = np.array(intersection.exterior.coords.xy)
            ptp = xy.ptp(axis=1)
            for move in [(ptp[0], 0), (-ptp[0], 0), 
                         (0, ptp[1]), (0, -ptp[1])]:
                temp_obj = shape_poly(obj.get_coords() + move)
                if type(temp_obj.intersection(wall_poly)) == shape_string:
                    obj.center += move
                    return
    else:
        distance = wall_poly.distance(obj_poly)
        overlaps = []
        dirs = [(distance, 0), (-distance, 0), (0, distance), (0, -distance)]
        for move in dirs:
            temp_obj = shape_poly(obj.get_coords() + move)
            if not temp_obj.intersects(wall_poly):
                overlaps.append(np.finfo(np.float).max)
            else:
                overlaps.append(obj_poly.intersection(wall_poly).area)
        snap_dir = np.argmin(np.array(overlaps))
        obj.center += dirs[snap_dir]
        return
    print('No wall is found to fit the criteria...')
           
import random

def gen_cube_obj(bbox, file_path, is_color=False, should_save=True):
    vertices = []
    a,b,c,d = bbox.get_coords()
    for x,y in [a,b,d,c]:
        vertices.append((x,y,bbox.z[1]))
    for x,y in [a,b,d,c]:
        vertices.append((x,y,bbox.z[0]))
    c=np.random.rand(3)
    faces = [(1,2,3),
             (2,4,3),
             (1,3,5),
             (3,7,5),
             (1,5,2),
             (5,6,2),
             (2,6,4),
             (6,8,4),
             (4,8,7),
             (7,3,4),
             (6,7,8),
             (6,5,7),
            ]
    faces = [(*f, -1) for f in faces]

    if should_save:
        with open(file_path, 'w') as fp:
            for v in vertices:
                if is_color:
                    v1,v2,v3 = v
                    fp.write('v {} {} {} {} {} {}\n'.format(v1, v2, v3, *c))
                else:
                    v1,v2,v3 = v
                    fp.write('v {} {} {}\n'.format(v1, v2, v3))

            for f in faces:
                fp.write('f {} {} {}\n'.format(*f[:-1]))

    return vertices, faces

def get_coords(edge_x, edge_y, center):
    '''
    Return the vertices of the bounding box, in order of BL,BR,TR,TL
    '''
    x = np.array(edge_x) / 2.
    y = np.array(edge_y) / 2.
    return np.array([center - x - y,
                     center + x - y,
                     center + x + y,
                     center - x + y])

def get_oriented_bbox(obj):
    bbox_x = np.linalg.norm(obj['edge_x'])
    bbox_y = np.linalg.norm(obj['edge_y'])
    theta = np.linalg.norm(obj['theta'])
    center = obj['center']
    vertices = []
    a, b, c, d = get_coords([1, 0], [0, 1], [0, 0])
    vertices = np.array([a, b, c, d])
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta), np.cos(theta)]])
    vertices *= [bbox_x, bbox_y]
    vertices = vertices.dot(rot)
    vertices += center
    return vertices
def get_triangle_area(p1, p2, p3):
    return np.abs((p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) / 2.0)
# check if the 2D point is inside the rotated 2D bounding box of the object
def check_point_in_obj_xy(obj, pt):
    bbox = get_oriented_bbox(obj)
    bbox_x = np.linalg.norm(obj['edge_x'])
    bbox_y = np.linalg.norm(obj['edge_y'])
    bbox_area = bbox_x * bbox_y
    triangles_area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        triangles_area += get_triangle_area(bbox[i], bbox[j], pt)
    return np.abs(triangles_area - bbox_area) < 1e-5

def has_overlap(bbox_i, bbox_j):
    coord_i_xy = shape_poly(bbox_i.get_coords())
    i_z = bbox_i.z
    coord_j_xy = shape_poly(bbox_j.get_coords())
    j_z = bbox_j.z
    if coord_i_xy.intersects(coord_j_xy):
        z_overlap = get_z_overlaps(i_z, j_z)
        return coord_i_xy.intersection(coord_j_xy).area * z_overlap
    return 0

# def has_overlap(obj1, obj2):
    # obj1_bbox = get_oriented_bbox(obj1)
    # obj2_bbox = get_oriented_bbox(obj2)
    # in_xy = False
    # for i in range(4):
        # if check_point_in_obj_xy(obj2, obj1_bbox[i]):
            # in_xy = True
    # for i in range(4):
        # if check_point_in_obj_xy(obj1, obj2_bbox[i]):
            # in_xy = True
    # obj1_z = obj1['z']
    # obj2_z = obj2['z']
    # in_z = obj1_z[0] < obj2_z[1] and obj2_z[0] < obj1_z[1]
    # return in_xy and in_z
def get_z_overlaps(z1, z2):
    bottom = max(z1[0], z2[0])
    top = min(z1[-1], z2[-1])
    return max(0, top - bottom)

def overlaps(bbox_i, bbox_j):
    coord_i_xy = shape_poly(bbox_i[1].get_coords())
    i_z = bbox_i[1].z
    coord_j_xy = shape_poly(bbox_j[1].get_coords())
    j_z = bbox_j[1].z
    if coord_i_xy.intersects(coord_j_xy):
        z_overlap = get_z_overlaps(i_z, j_z)
        return coord_i_xy.intersection(coord_j_xy).area * z_overlap
    return 0

def get_volume(bbox):
    x,y,z = bbox.get_scale()
    return x*y*z
