import numpy as np
# import poly_decomp as pd
from . import earcut

from .utils import BBox

class Wall_bbox(object):
    
    def __init__(self, bbox, ceiling_height=2.4):
        self.t = ceiling_height
        self.b = 0.
        # self.bbox = BBox(raw_pts, self.b, self.t)
        self.bbox = bbox

        x_scale,y_scale,_ = self.bbox.get_scale()

    
        self.theta = self.bbox.get_rotation_angle(CAD_dir=(1,0))
        theta = -self.theta
        # print(theta, self.bbox.get_rotation_angle(CAD_dir=(1,0)))
        self.rot_2d = np.array([[np.cos(theta),-np.sin(theta)],
                         [np.sin(theta),np.cos(theta)]])

        self.rot_3d = np.array([[np.cos(theta),-np.sin(theta),0],
                         [np.sin(theta),np.cos(theta),0],
                         [0,0,1]])

        straight_pts = self.bbox.get_coords().dot(self.rot_2d)  
        xmin, xmax = straight_pts[:,1].min(), straight_pts[:,1].max()
        ymin, ymax = straight_pts[:,0].min(), straight_pts[:,0].max()

        if (xmax - xmin) > (ymax - ymin):
            self.x_principle = True
            self.l = xmin
            self.r = xmax
            self.f = ymin
            self.back = ymax
        else:
            self.x_principle = False
            self.l = ymin
            self.r = ymax
            self.f = xmin
            self.back = xmax

        # doors are incorporated as bottom_edge polygons intrusion
        # some windows (or cut-outs) are top_edge polygons intrusion
        # similarly for side edges
        self.bottom_edge_poly = []
        self.top_edge_poly = []
        self.right_edge_poly = []
        self.left_edge_poly = []

        self.tl_corner = None
        self.bl_corner = None
        self.tr_corner = None
        self.br_corner = None

        self.holes = []   
        self.vns = {'x_minus' : 1, 'x_plus' : 2, 
                    'y_minus' : 3, 'y_plus' : 4,
                    'z_minus' : 5, 'z_plus' : 6}
    
    # def add_hole(self, b, t, raw_pts):
    def add_hole(self, bbox):
        straight_pts = bbox.get_coords().dot(self.rot_2d)  
        x_scale, y_scale,_ = bbox.get_scale()
        b,t = bbox.z
        
        if self.x_principle:
            l,r = straight_pts[:,1].min(), straight_pts[:,1].max()
        else:
            l,r = straight_pts[:,0].min(), straight_pts[:,0].max()

        if b < 0 or  t > self.t:
            if b < 0:
                b = 0.05
            if t > self.t:
                t = self.t - 0.05
        if l < self.l or r > self.r:
            if l < self.l:
                l = self.l + 0.01
            if r > self.r:
                r = self.r - 0.01
            edge_new = l-r
            # center_new = (l+r) / 2.
            if self.x_principle:
                straight_pts[:,1] = np.clip(straight_pts[:,1], 
                                            self.l + 0.01, self.r-0.01)
            else:
                straight_pts[:,0] = np.clip(straight_pts[:,0], 
                                            self.l + 0.01, self.r-0.01)

            if x_scale > y_scale:
                bbox.edge_x = bbox.edge_x / x_scale * edge_new
            else:
                bbox.edge_y = bbox.edge_y / y_scale * edge_new
            # print(self.l, self.r, self.back, self.f)
            # raise ValueError('Window coordinate out of range... {}'.format(
                            # (self.t, self.l, self. r, l, r, b, t)))
        isb, ist, isl, isr = b == 0, t == self.t, l == self.l, r == self.r
        if (isb and ist) or (isl and isr):
            raise ValueError('Wall broken into two components, not supported')
        if not (isb or ist or isl or isr):
            self.holes.append((b,t,l,r))
        elif ist and isl:
            if self.tl_corner is not None:
                raise ValueError('top left corner already occupied')
            self.tl_corner = (b,t,l,r)
        elif ist and isr:
            if self.tr_corner is not None:
                raise ValueError('top right corner already occupied')
            self.tr_corner = (b,t,l,r)
        elif isb and isl:
            if self.bl_corner is not None:
                raise ValueError('bottom left corner already occupied')
            self.bl_corner = (b,t,l,r)
        elif isb and isr:
            if self.br_corner is not None:
                raise ValueError('bottom right corner already occupied')
            self.br_corner = (b,t,l,r)
        elif ist:
            self.top_edge_poly.append((b,t,l,r))
        elif isb:
            self.bottom_edge_poly.append((b,t,l,r))
        elif isl:
            self.left_edge_poly.append((b,t,l,r))
        elif isr:
            self.right_edge_poly.append((b,t,l,r))
        else:
            raise ValueError('Logic is wrong, the hole is not being added')


        # print('######################')
        # print('original: ', bbox.center, bbox.edge_x, bbox.edge_y)

        straight_pts_center = straight_pts.mean(axis=0)
        thickness = (self.f - self.back) 
        # print('thickness: ', thickness)
        if self.x_principle:
            straight_pts_center[0] = (self.back + self.f) / 2.
        else:
            straight_pts_center[1] = (self.back + self.f) / 2.
        if x_scale < y_scale:
            bbox.edge_x = bbox.edge_x / x_scale * thickness
        else:
            bbox.edge_y = bbox.edge_y / y_scale * thickness
        # print('intermediate: ', bbox.center, bbox.edge_x, bbox.edge_y)
        bbox.center = straight_pts_center.dot(np.linalg.inv(self.rot_2d))
        # print('final: ', bbox.center, bbox.edge_x, bbox.edge_y)
        return bbox
        
    def build(self):
        self.bottom_edge_poly.sort(key=lambda x: x[2])
        self.top_edge_poly.sort(key=lambda x: -x[3])
        self.right_edge_poly.sort(key=lambda x: x[0])
        self.left_edge_poly.sort(key=lambda x: -x[0])
        
        f_vertices, f_faces = self.get_plane(front=True)
        b_vertices, b_faces = self.get_plane(front=False)
        b_faces[:,:3] += len(f_vertices)
        vertices = np.concatenate([f_vertices, b_vertices], axis=0)
        faces = np.concatenate([f_faces, b_faces], axis=0)

        self.vertices = vertices
        cross_faces = self.build_front_back_faces()
        faces = np.concatenate([cross_faces, faces], axis=0)
        self.faces = faces
        return vertices.dot(np.linalg.inv(self.rot_3d)), faces    

    # def build_collision_mesh(self):
        # self.get_convex_decomposed_plane()

    def build_collision_mesh(self, f_path):
        vertices, holes = self.get_plane_vertices()  
        faces = np.array(earcut.earcut(vertices, holes))
        faces = faces.reshape((-1, 3))

        vertices = np.array(vertices).reshape((-1, 2))
        decomposed = []
        for f1,f2,f3 in faces:
            decomposed.append([vertices[f1,:],vertices[f2,:],vertices[f3,:]])
        # polygon = []
        # for i in range(len(vertices)):
            # v = vertices[-(i+1), :]
            # polygon.append((v[0], v[1]))

        verts = []; fs = []
        for poly in decomposed:
            poly_np = np.array(poly)
            poly_np_mean = np.mean(poly_np, axis=0)
            poly_np = poly_np_mean + (poly_np - poly_np_mean) * 0.999
            flattened = poly_np.reshape(-1)
            pf = np.array(earcut.earcut(flattened)) + 1
            pf= pf.reshape((-1, 3))
            bf = pf[:,:] + len(poly_np)

            # front 
            f_z = [self.f,] * int(poly_np.shape[0])

            if not self.x_principle:
                front_v = np.vstack([poly_np[:, 1], f_z,
                                    poly_np[:, 0]]).transpose()
            else:
                front_v = np.vstack([f_z, poly_np[:, 1],  
                                    poly_np[:, 0]]).transpose()
            if not self.x_principle:
                pf = np.vstack([pf[:,0], pf[:,2], pf[:,1]]).transpose()
            pf = np.append(pf, np.array([0,] * len(pf))[:, None], axis=1)
            #front_v, pf

            # back
            b_z = [self.back,] * int(poly_np.shape[0])
            if not self.x_principle:
                back_v = np.vstack([poly_np[:, 1],  b_z,
                                    poly_np[:, 0]]).transpose()
            else:
                back_v = np.vstack([b_z,poly_np[:, 1],  
                                    poly_np[:, 0]]).transpose()
            if self.x_principle:
                bf = np.vstack([bf[:,0], bf[:,2], bf[:,1]]).transpose()
            bf = np.append(bf, np.array([0,] * len(bf))[:, None], axis=1)
            #back_v, bf

            # front-back
            num_front_plane = len(front_v)
            cross_faces = []
            curr_point = 1
            for idx in range(num_front_plane):
                if idx == num_front_plane- 1:
                    next_point = 1
                else:
                    next_point = curr_point + 1
                if not self.x_principle:
                    cross_faces.append((curr_point, curr_point + num_front_plane, 
                                        next_point, self.vns['z_minus']))
                    cross_faces.append((next_point, curr_point + num_front_plane, 
                                        next_point + num_front_plane, self.vns['z_minus']))
                else:
                    cross_faces.append((curr_point, next_point,
                                        curr_point + num_front_plane, 
                                        self.vns['z_minus']))
                    cross_faces.append((next_point, next_point + num_front_plane,
                                        curr_point + num_front_plane, self.vns['z_minus']))
                curr_point = next_point 

            verts.append( np.concatenate([front_v, back_v], axis=0) )
            cross_faces = np.asarray(cross_faces)
            fs.append( np.concatenate([pf, bf, cross_faces], axis=0) )
        # vertices = np.concatenate(verts, axis=0)
        with open(f_path, 'w') as fp:
            total_v = 0
            for idx, (v,f) in enumerate(zip(verts,fs)):
                f = f + total_v
                total_v += len(v)
                fp.write('o convex_{}\n'.format(idx))
                v= v.dot(np.linalg.inv(self.rot_3d))
                for vs in v:
                    fp.write('v {} {} {}\n'.format(*vs))

                for f1,f2,f3,fn in f:
                    fp.write('f {f1} {f2} {f3}\n'.format(f1=f1,f2=f2,f3=f3))



    def get_plane(self, front=True):
        ### process front
        vertices, holes = self.get_plane_vertices()  
        z_val = self.f if front else self.back
        vertices_z = [z_val,] * int( len(vertices) / 2)
        
        faces = np.array(earcut.earcut(vertices, holes)) + 1
        faces = faces.reshape((-1, 3))
        vertices = np.array(vertices).reshape((-1, 2))

        if not self.x_principle:
            vertices = np.vstack([vertices[:, 1],  
                                vertices_z,
                                vertices[:, 0]]).transpose()
            f_normal = self.vns['y_minus'] if front else self.vns['y_plus']
        else:
            vertices = np.vstack([vertices_z,
                                vertices[:, 1], 
                                vertices[:, 0]]).transpose()
            f_normal = self.vns['x_minus'] if front else self.vns['x_plus']
        if (self.x_principle and not front) or (not self.x_principle and front):
            faces = np.vstack([faces[:,0], faces[:,2], faces[:,1]]).transpose()
        faces = np.append(faces, np.array([f_normal,] * len(faces))[:, None], axis=1)

        return vertices, faces
    
        
    def build_front_back_faces(self):
        num_front_plane = int(self.vertices.shape[0] / 2)
        cross_faces = []
        
        left_normal = self.vns['x_minus'] if self.x_principle else self.vns['y_minus']
        right_normal = self.vns['x_plus'] if self.x_principle else self.vns['y_plus']

        curr_point = 1
        total_face = 4
        if self.bl_corner is not None:
            total_face += 2
        if self.br_corner is not None:
            total_face += 2
        if self.tr_corner is not None:
            total_face += 2
        if self.tl_corner is not None:
            total_face += 2
            
        total_face += len(self.bottom_edge_poly) * 4
        total_face += len(self.right_edge_poly) * 4
        total_face += len(self.top_edge_poly) * 4
        total_face += len(self.left_edge_poly) * 4
            
        for idx in range(total_face):
            if idx == total_face - 1:
                next_point = 1
            else:
                next_point = curr_point + 1
            if not self.x_principle:
                cross_faces.append((curr_point, curr_point + num_front_plane, 
                                    next_point, self.vns['z_minus']))
                cross_faces.append((next_point, curr_point + num_front_plane, 
                                    next_point + num_front_plane, self.vns['z_minus']))
            else:
                cross_faces.append((curr_point, next_point,
                                    curr_point + num_front_plane, 
                                    self.vns['z_minus']))
                cross_faces.append((next_point, next_point + num_front_plane,
                                    curr_point + num_front_plane, self.vns['z_minus']))
            curr_point = next_point 
        
        curr_point = num_front_plane - len(self.holes) * 4 + 1
        
        for idx in range(len(self.holes)):
            curr_holes_start = curr_point
            for i in range(4):
                if i == 3:
                    next_point = curr_holes_start
                else:
                    next_point = curr_point + 1
                if not self.x_principle:
                    cross_faces.append((curr_point, curr_point + num_front_plane, 
                                        next_point, self.vns['z_minus']))
                    cross_faces.append((next_point, curr_point + num_front_plane, 
                                        next_point + num_front_plane, self.vns['z_minus']))
                else:
                    cross_faces.append((curr_point, next_point,
                                        curr_point + num_front_plane, 
                                        self.vns['z_minus']))
                    cross_faces.append((next_point, next_point + num_front_plane,
                                        curr_point + num_front_plane, self.vns['z_minus']))
                curr_point = next_point 
            curr_point += 4
            
        
        return np.asarray(cross_faces)

    def get_plane_vertices(self):
        vertices = []
        if self.bl_corner is not None:
            b,t,l,r = self.bl_corner
            vertices.extend([t, l, t, r, b, r])
            vertices_z.extend([self.f])
        else:
            vertices.extend([self.b,self.l])
        for b,t,l,r in self.bottom_edge_poly:
            vertices.extend([b, l, t, l, t, r, b, r])
        if self.br_corner is not None:
            b,t,l,r = self.br_corner
            vertices.extend([b, l, t, l, t, r])
        else:
            vertices.extend([self.b,self.r])
        for b,t,l,r in self.right_edge_poly:
            vertices.extend([b, r, b, l, t, l, t, r])
        if self.tr_corner is not None:
            b,t,l,r = self.tr_corner
            vertices.extend([b, r, b, l, t, l])
        else:
            vertices.extend([self.t,self.r])
        for b,t,l,r in self.top_edge_poly:
            vertices.extend([t, r, b, r, b, l, t, l])
        if self.tl_corner is not None:
            b,t,l,r = self.tl_corner
            vertices.extend([t, r, b, r, b, l])
        else:
            vertices.extend([self.t,self.l])
        for b,t,l,r in self.left_edge_poly:
            vertices.extend([t, l, t, r, b, r, b, l])
        
        holes = []

        for b,t,l,r in self.holes:
            holes.append(int(len(vertices) / 2))
            vertices.extend([b, l, t, l, t, r, b, r])
        return vertices, holes

    def add_skirt(self, skirt_pts):
        num_pts = skirt_pts.shape[0]
        skirt_verts = []
        skirt_faces = []
        
        segments = []
        if self.bl_corner is not None:
            b,t,l,r = self.bl_corner
            segments.append(r )
        else:
            segments.append(self.l)
        
        for b,t,l,r in self.bottom_edge_poly:
            segments.extend([l, r])
        
        if self.br_corner is not None:
            b,t,l,r = self.br_corner
            segments.append(l)
        else:
            segments.append(self.r)
        
        # front
        segments = np.array(segments).reshape([-1, 2])
        num_seg = len(segments)
        for front in [True, False]:
            reverse_factor = -1 if front else 1
            x_loc = self.f if front else self.back
            for idx, (seg_start, seg_end) in enumerate(segments):
                if not front:
                    seg_start, seg_end = seg_end, seg_start
                for x, z in skirt_pts:
                    skirt_verts.append((x_loc + reverse_factor * x, seg_start + reverse_factor * x, z) 
                                       if self.x_principle else 
                                       (seg_start + reverse_factor * x, x_loc + reverse_factor * x, z))
                for x, z in skirt_pts:
                    skirt_verts.append((x_loc + reverse_factor * x, seg_end - reverse_factor * x, z) 
                                       if self.x_principle else 
                                       (seg_end - reverse_factor * x, x_loc + reverse_factor * x, z))
                vert_idx_offset = idx * num_pts * 2
                if not front:
                    vert_idx_offset += num_seg * num_pts * 2
                for i in range(num_pts):
                    curr_point = i + vert_idx_offset + 1
                    if i == num_pts - 1:
                        next_point = vert_idx_offset + 1
                    else:
                        next_point = curr_point + 1
                    if self.x_principle:
                        skirt_faces.append((curr_point, next_point, 
                                            curr_point + num_pts, self.vns['z_minus']))
                        skirt_faces.append((next_point, next_point + num_pts,
                                            curr_point + num_pts, self.vns['z_minus']))
                    else:
                        skirt_faces.append((curr_point, curr_point + num_pts, 
                                            next_point, self.vns['z_minus']))
                        skirt_faces.append((next_point, curr_point + num_pts,
                                            next_point + num_pts, self.vns['z_minus']))

        
        num_front = len(skirt_verts)
        left = self.bl_corner[-1] if self.bl_corner is not None else self.l
        right = self.br_corner[-2] if self.br_corner is not None else self.r
        segments = [(True, left, self.f, self.back), (False, right, self.back, self.f)]
        
        for minus, pos, seg_start, seg_end in segments:
            reverse_factor = -1 if minus else 1

            for x, z in skirt_pts:
                skirt_verts.append((pos + reverse_factor * x, seg_start + reverse_factor * x, z) 
                                   if not self.x_principle else 
                                   (seg_start + reverse_factor * x, pos + reverse_factor * x, z))
            for x, z in skirt_pts:
                skirt_verts.append((pos + reverse_factor * x, seg_end - reverse_factor * x, z) 
                                   if not self.x_principle else 
                                   (seg_end - reverse_factor * x, pos + reverse_factor * x, z))
            vert_idx_offset = (1 - int (minus)) * num_pts * 2 + num_front
            for i in range(num_pts):
                curr_point = i + vert_idx_offset + 1
                if i == num_pts - 1:
                    next_point = vert_idx_offset + 1
                else:
                    next_point = curr_point + 1
                if not self.x_principle:
                    skirt_faces.append((curr_point, next_point, 
                                        curr_point + num_pts, self.vns['z_minus']))
                    skirt_faces.append((next_point, next_point + num_pts,
                                        curr_point + num_pts, self.vns['z_minus']))
                else:
                    skirt_faces.append((curr_point, curr_point + num_pts, 
                                        next_point, self.vns['z_minus']))
                    skirt_faces.append((next_point, curr_point + num_pts,
                                        next_point + num_pts, self.vns['z_minus']))                
        return np.array(skirt_verts).dot(np.linalg.inv(self.rot_3d)), np.array(skirt_faces)
