from xml.dom import minidom

from shapely.geometry import Polygon as shape_poly
from shapely.geometry import LineString as shape_string
import numpy as np
import random
import os
import open3d as o3d

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return (ang2 - ang1) % (2 * np.pi)

# def getCenter(p):
    # return reduce((lambda x, y: x + y), p) / len(p)

# def getShapeRotationVertex(pts):
    # if pts[0].x() > pts[1].x(): id1, id2 = 1, 0
    # else: id1, id2 = 0, 1

    # if len(pts) >= 4:
        # eucl = lambda a : (a.x()**2. + a.y()**2.)**.5
        # vec_points = pts[id1] - pts[id2]
        # dist_points = eucl(vec_points)
        # if dist_points > 0:
            # # compute size of the shape
            # d = self.point_size / self.scale
            # des_len = 5 * d
            # # compute the normal vector and add it to the center of the two points
            # nrm = QPointF(-(vec_points.y() * des_len / dist_points),
                         # vec_points.x() * des_len / dist_points)
            # center_shape = self.getCenter()
            # center_points = (pts[id1] + pts[id2])/2
            # # find the candidate that lies farest from the center
            # candidate1 = center_points + nrm
            # candidate2 = center_points - nrm
            # d1 = eucl(candidate1 - center_shape)
            # d2 = eucl(candidate2 - center_shape)
            # c2 = d1 < d2
            # return (candidate2 if c2 else candidate1), d, c2, center_shape
    # return None

        # if self.pts[0,0] > self.pts[1,0]: 
            # # self.origin = 0 if self.flip_factor == 1 else 3
            # self.origin = self.pts[0 if self.flip_factor == 1 else 3]
            # self.edge_x = self.pts[1, :] - self.pts[0, :]
            # self.edge_y = self.flip_factor*(self.pts[3, :] - self.pts[0, :])
        # else:
            # self.origin = self.pts[1 if self.flip_factor == 1 else 2]
            # self.edge_x = self.pts[0, :] - self.pts[1, :]
            # self.edge_y = self.flip_factor*(self.pts[3, :] - self.pts[0, :])

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
    # print(bbox.get_scale())

    # x_range = xmax-xmin
    # y_range = ymax-ymin
    # if x_range < y_range:
        # x_mean = (xmin+xmax)/2.
        # return x_mean-size/2.,x_mean+size/2.,ymin,ymax
    # else:
        # y_mean = (ymin+ymax)/2.
        # return xmin,xmax,y_mean-size/2.,y_mean+size/2.

def get_unique(doc, val):
    it = doc.getElementsByTagName(val)
    if len(it) > 1:
        raise ValueError('value not unique...')
    return it[0].firstChild.nodeValue


def snap_out_of_wall(obj, wall, obj_poly, wall_poly):
    # wall_bbox = wall[:4]
    # xmin,xmax,ymin,ymax = wall_bbox
    # wall_poly = shape_poly(list(zip([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax])))
    
    # obj_bbox = obj[1:5]
    # xmin,xmax,ymin,ymax = obj_bbox
    # obj_poly = shape_poly(list(zip([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax])))

    intersection = wall_poly.intersection(obj_poly)
    if type(intersection) == shape_string:
        overlap_margin = 0.001
        for move in [(overlap_margin, 0), (-overlap_margin, 0), 
                     (0, overlap_margin), (0, -overlap_margin)]:
            # obj_poly = shape_poly(np.array(list(zip([xmin, xmax, xmax, xmin], 
                                               # [ymin, ymin, ymax, ymax]))) + move)
            temp_obj = shape_poly(obj.get_coords() + move)
            if not temp_obj.intersects(wall_poly):
                obj.center += move
                return
                # return (obj[0],xmin+move[0],xmax+move[0],
                        # ymin+move[1],ymax+move[1],obj[5],obj[6],obj[7])
        
    elif type(intersection) == shape_poly:
        xy = np.array(intersection.exterior.coords.xy)
        #print(x_size,y_size, obj)
        ptp = xy.ptp(axis=1) + 0.001
        for move in [(ptp[0], 0), (-ptp[0], 0), 
                     (0, ptp[1]), (0, -ptp[1])]:
            temp_obj = shape_poly(obj.get_coords() + move)
            if not temp_obj.intersects(wall_poly):
                obj.center += move
                return

def snap_on_wall(obj, wall, obj_poly, wall_poly):
    # wall_bbox = wall[:4]
    # xmin,xmax,ymin,ymax = wall_bbox
    # wall_poly = shape_poly(list(zip([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax])))
    
    # obj_bbox = obj[1:5]
    # xmin,xmax,ymin,ymax = obj_bbox
    # obj_poly = shape_poly(list(zip([xmin, xmax, xmax, xmin], [ymin, ymin, ymax, ymax])))
    if wall_poly.intersects(obj_poly):
        intersection = wall_poly.intersection(obj_poly)
        if type(intersection) == shape_string:
            return
        elif type(intersection) == shape_poly:
            xy = np.array(intersection.exterior.coords.xy)
            #print(x_size,y_size, obj)
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
    # ymin,ymax,xmin,xmax,zmin,zmax = bbox
    # vertices = [(xmin,ymin,zmin),
                # (xmin,ymin,zmax),
                # (xmin,ymax,zmin),
                # (xmin,ymax,zmax),
                # (xmax,ymin,zmin),
                # (xmax,ymin,zmax),
                # (xmax,ymax,zmin),
                # (xmax,ymax,zmax)]
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
            # for v in vertices:
                # fp.write('v {} {} {}\n'.format(*v))

            for f in faces:
                fp.write('f {} {} {}\n'.format(*f[:-1]))

    return vertices, faces



def align_CAD_to_bbox(mesh_path, orient_box, articulation_root=None, save_dir=None, idx=0):
    if articulation_root != None:
        mesh_dir = mesh_path
        mesh_path = os.path.join(mesh_dir, articulation_root)
        meshes = [f for f in os.listdir(mesh_dir) if f[-4:] == '.obj' and f != articulation_root]
    else:
        meshes = []

    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mverts = np.asarray(mesh.vertices)
    mfaces = np.array(mesh.triangles) + 1
    mesh.clear()

    #scale 
    box_size = mverts.ptp(axis=0)
    x,y,z = orient_box.get_scale()
    scale = np.array([x,y,z]) / box_size
    mverts = mverts * scale

    # rotation
    theta = orient_box.get_rotation_angle()
    rotation = np.array([[np.cos(theta),-np.sin(theta),0],
                         [np.sin(theta),np.cos(theta),0],
                         [0,0,1]])
    mverts = mverts.dot(rotation)

    # translate
    box_min = mverts.min(axis=0)
    # box_min = box_min.dot(rotation)
    x,y  = (orient_box.center - 
            np.abs(orient_box.edge_x)/2-
            np.abs(orient_box.edge_y)/2)
    translate = np.array([x,y,orient_box.z[0]]) - box_min
    mverts = mverts + translate 
    
    
    if save_dir is not None:
        save_path = os.path.join(save_dir, '{}_{}'.format(idx, os.path.basename(mesh_path)))
        with open(save_path, 'w') as fp:
            for v in mverts:
                fp.write('v {} {} {}\n'.format(*v))
            for f in mfaces:
                fp.write('f {} {} {}\n'.format(*f))
        for m in meshes:
            mesh = o3d.io.read_triangle_mesh(os.path.join(mesh_dir, m))
            mverts = np.asarray(mesh.vertices)
            mfaces = np.array(mesh.triangles) + 1
            mesh.clear()

            mverts = (mverts * scale).dot(rotation) + translate
            save_path = os.path.join(save_dir, '{}_{}'.format(idx, m))
            with open(save_path, 'w') as fp:
                for v in mverts:
                    fp.write('v {} {} {}\n'.format(*v))
                for f in mfaces:
                    fp.write('f {} {} {}\n'.format(*f))
    return scale, rotation, translate
            
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

def has_overlap(obj1, obj2):
    obj1_bbox = get_oriented_bbox(obj1)
    obj2_bbox = get_oriented_bbox(obj2)
    in_xy = False
    for i in range(4):
        if check_point_in_obj_xy(obj2, obj1_bbox[i]):
            in_xy = True
    for i in range(4):
        if check_point_in_obj_xy(obj1, obj2_bbox[i]):
            in_xy = True
    obj1_z = obj1['z']
    obj2_z = obj2['z']
    in_z = obj1_z[0] < obj2_z[1] and obj2_z[0] < obj1_z[1]
    return in_xy and in_z

def get_volume(bbox):
    x,y,z = bbox.get_scale()
    return x*y*z
