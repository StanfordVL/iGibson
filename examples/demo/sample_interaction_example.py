import gibson2
from gibson2.simulator import Simulator
from gibson2.render.viewer import Viewer
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.profiler import Profiler
from gibson2.utils.assets_utils import get_ig_scene_path
import argparse
import os
from PIL import Image
import numpy as np
import subprocess
import random

from threading import Thread
import logging

import cv2
import pybullet as p
from gibson2.objects.visual_marker import VisualMarker
from gibson2.utils.utils import rotate_vector_2d
import time
import math
import pdb
import json
from tqdm import tqdm

FETCH_HEIGHT=1.08
INTERACTION_TARGET=0.3
DEPTH_HIGH = 20.

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, 
                        help='Name of the scene in the iG Dataset')
    parser.add_argument('--save_dir', type=str, help='Directory to save the frames.',
                        default='misc/interaction_pretrain')
    parser.add_argument('--resolution', type=int, default=512, 
                        help='Image resolution.')
    parser.add_argument('--samples', type=int, default=4000, 
                        help='number of sampled locations')
    parser.add_argument('--interactions', type=int, default=10, 
                        help='number of interactions')
    parser.add_argument('--seed', type=int, default=15, 
                        help='Random seed.')
    parser.add_argument('--domain_rand', dest='domain_rand',
                        action='store_true')
    parser.add_argument('--domain_rand_interval', dest='domain_rand_interval',
                        type=int, default=50)
    parser.add_argument('--object_rand', dest='object_rand',
                        action='store_true')
    args = parser.parse_args()
    return args

def init_scene(scene):
    body_joint_pairs = scene.open_all_objs_by_categories(
            ['bottom_cabinet',
             'bottom_cabinet_no_top',
             'top_cabinet',
             'dishwasher',
             'fridge',
             'microwave',
             'oven',
             'washer'
             'dryer',
             'door',
             ], mode='random')

class InteractionSampler(Viewer):
    def __init__(self,
                 simulator,
                 renderer,
                 initial_pos=[0, 0, 1.2],
                 initial_view_direction=[1, 0, 0],
                 initial_up=[0, 0, 1],
                 min_cam_z=-1e6,
                 headless=True,
                 modes=('rgb', '3d', 'seg')
                 ):
        self.px = initial_pos[0]
        self.py = initial_pos[1]
        self.pz = initial_pos[2]
        self.theta = np.arctan2(
            initial_view_direction[1], initial_view_direction[0])
        self.phi = np.arctan2(initial_view_direction[2], 
                              np.sqrt(initial_view_direction[0] ** 2 +
                              initial_view_direction[1] ** 2))
        self.min_cam_z = min_cam_z

        self.view_direction = np.array(initial_view_direction)
        self.up = initial_up

        self.renderer = renderer
        self.simulator = simulator
        self.modes = modes
        self.cid = []
        
        self.headless = headless
        if not headless:
            cv2.namedWindow('ExternalView')
            cv2.moveWindow("ExternalView", 0, 0)
        self.create_visual_object()

    def raycast(self, x, y, ray_distance=20.):
        camera_pose = np.array([self.px, self.py, self.pz])
        self.renderer.set_camera(
            camera_pose, camera_pose + self.view_direction, self.up)
        position_cam = np.array([(x - self.renderer.width / 2) / float(self.renderer.width / 2) * np.tan(
            self.renderer.horizontal_fov / 2.0 / 180.0 * np.pi),
            -(y - self.renderer.height / 2) / float(self.renderer.height / 2) * np.tan(
            self.renderer.vertical_fov / 2.0 / 180.0 * np.pi),
            -1,
            1])
        position_cam[:3] *= ray_distance
        position_world = np.linalg.inv(self.renderer.V).dot(position_cam)
        position_eye = camera_pose
        res = p.rayTest(position_eye, position_world[:3])
        return res
    
    def set_pose(self, loc, view_dir, up=[0,0,1]):
        self.x,self.y,self.z = loc
        self.view_direction = view_dir
        self.up = up

    def move_constraint_3d(self, position_world):
        self.constraint_marker.set_position(position_world)
        self.constraint_marker2.set_position(position_world)

    def world2cam(self, position_world):
        cam_space = self.renderer.V.dot([*position_world, 1])
        image_space = self.renderer.get_intrinsics().dot(cam_space[:-1])
        x,y = (image_space[:-1] / image_space[-1]).astype(int)
        return (self.renderer.width - x,y)

    def change_dir(self, event, x, y, flags, param):
        return

    def update(self):
        camera_pose = np.array([self.px, self.py, self.pz])
        self.renderer.set_camera(
            camera_pose, camera_pose + self.view_direction, self.up)
        outputs = self.renderer.render(modes=self.modes)
        if not self.headless:
            # assume mode is rgb,3d,seg
            colors = np.array([(241,187,123),(253,100,103),
                      (91,26,24),(214,114,54),
                      (230,160,196),(198,205,247),
                      (216,164,153),(114,148,212)])
            seg = (outputs[-1][:,:,0] * 255).astype(int)
            seg_color = colors[seg.reshape(-1) % len(colors)].astype(float) / 255.
            outputs[-1][:,:,:-1] = seg_color.reshape((*seg.shape, 3))
            frame = cv2.cvtColor(np.concatenate(outputs, axis=1),
                                 cv2.COLOR_RGB2BGR)
            cv2.imshow('ExternalView', frame)
            cv2.waitKey(1)
        return_vals = dict(zip(self.modes, outputs))
        if 'rgb' in return_vals:
            return_vals['rgb'] = (return_vals['rgb'][:,:,:3]*255).astype(np.uint8)
        if '3d' in return_vals:
            depth = -return_vals['3d'][:,:,2:3]
            depth[depth > DEPTH_HIGH] = DEPTH_HIGH
            depth[depth < 0.] = 0.
            return_vals['3d'] =((depth / DEPTH_HIGH) * np.iinfo(np.uint16).max).astype(np.uint16)
        if 'seg' in return_vals:
            return_vals['seg'] = (return_vals['seg'][:,:,0]*255).astype(np.uint8)
        cam = {}
        cam['cam_pose'] = (self.px, self.py, self.pz)
        cam['cam_dir'] = tuple(self.view_direction)
        cam['cam_up'] = tuple(self.up)
        return return_vals,cam


def main():
    args = parse_args()
    hdr_texture1 = os.path.join(
                 gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
    hdr_texture2 = os.path.join(
                 gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    light_map= os.path.join(
                 get_ig_scene_path(args.scene), 'layout', 'floor_lighttype_0.png')

    background_texture = os.path.join(
                gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

    settings = MeshRendererSettings(
               env_texture_filename=hdr_texture1,
               env_texture_filename2=hdr_texture2,
               env_texture_filename3=background_texture,
               light_modulation_map_filename=light_map,
               enable_shadow=True, msaa=True,
               skybox_size=36.,
               light_dimming_factor=0.8)
    
    s = Simulator(mode='headless', 
            image_width=args.resolution, image_height=args.resolution, 
            vertical_fov=90, rendering_settings=settings
            )

    random.seed(args.seed)
    scene = InteractiveIndoorScene(
            args.scene, texture_randomization=args.domain_rand,
            object_randomization=args.object_rand)

    s.import_ig_scene(scene)
    init_scene(scene)

    save_dir = os.path.join(get_ig_scene_path(args.scene), args.save_dir)
    os.makedirs(save_dir, exist_ok=True)

    for _ in range(60):
        s.step()
    s.sync()


    interactor = InteractionSampler(
                 simulator=s,
                 renderer=s.renderer,
                 initial_pos=[0, 0, FETCH_HEIGHT],
                 initial_view_direction=[1, 0, 0],
                 initial_up=[0, 0, 1],
                 headless=True)

    random.seed(8)
    np.random.seed(8)
    for sample_i in tqdm(range(args.samples)):
        if args.domain_rand and sample_i % args.domain_rand_interval == 0:
            scene.randomize_texture()

        # sample camera location
        _,floor_pt= scene.get_random_point() 
        standat = floor_pt[:-1]
        lookat = np.random.random((2,)) + 1e-6
        lookat = lookat / np.linalg.norm(lookat)

        recorded_data = [] 
        # interact for multiple steps
        for interaction_i in range(args.interactions):
            curr = {}
            # pre-render views
            interactor.set_pose([*standat, FETCH_HEIGHT],
                                [*lookat, 0.])

            curr['imgs_pre'],curr['cam'] = interactor.update()

            # sample pixel location
            pix_x = math.floor(random.random() * s.renderer.width)
            pix_y = math.floor(random.random() * s.renderer.height)
            if interaction_i % 3 == 0:
                # for every 4 samples, we want at least one that interacts with object
                non_structure = list(zip(*np.where(curr['imgs_pre']['seg'] > 2)))
                if len(non_structure) > 0:
                    pix_y, pix_x = random.choice(non_structure)
                    pix_y = int(pix_y)
                    pix_x = int(pix_x)

            curr['interact_at'] = (pix_x, pix_y)
            curr['object_cat'] = int(curr['imgs_pre']['seg'][pix_y,pix_x])
            # print('interacting at : {},{}'.format(pix_x, pix_y))

            # retrieve joint / link information
            res = interactor.raycast(pix_x, pix_y)
            # if ray misses any object 
            # (maybe outside of a window)
            if len(res) == 0 or res[0][0] == -1:
                # interaction fails, nothing happens
                curr['imgs_post'] = None 
                curr['interaction_pre']= None
                curr['interaction_post']= None
                curr['hit']= None
                recorded_data.append(curr)
                continue
            object_id, link_id, _, hit_pos, hit_normal = res[0]
            object_name = ''
            if link_id != -1:
                if object_id in scene.objects_by_id:
                    object_name = scene.objects_by_id[object_id].model
                else:
                    object_name = 'OBJECT_ID_NOT_FOUND'


            curr['hit']= {'object_id':object_id, 
                          'link_id':link_id, 
                          'object_name':object_name, 
                          'pos':tuple(hit_pos), 
                          'normal':tuple(hit_normal)} 
            cam_back = interactor.world2cam(hit_pos)

            interaction_pre = {'joint':None, 
                    'link':get_link_pose(object_id, link_id),
                    'constraint':tuple(hit_pos)}
            fixed = False # if it's the base link, we use p2p constraint
            if link_id != -1:
                joint_info = get_joint_info(object_id, link_id)
                joint_info['pos'] = p.getJointState(object_id, link_id)[0]
                interaction_pre['joint'] = joint_info
                if joint_info['type'] != p.JOINT_REVOLUTE:
                    # only for revolute joint, we use p2p constraint
                    fixed = True 

            # interact pix_loc
            interactor.create_constraint(pix_x, pix_y, fixed,
                                         ray_distance=20.,max_force=60.)
            s.step()
            hit_target = (np.array(hit_pos) - 
                          np.array(hit_normal) * INTERACTION_TARGET)
            interactor.move_constraint_3d(hit_target)
            for _ in range(20):
                s.step()
            interactor.remove_constraint()
            s.step()

            # render result after interaction:
            curr['imgs_post'],_ = interactor.update()
            interaction_post = {'joint':None, 
                    'link':get_link_pose(object_id, link_id),
                    'constraint':tuple(hit_target)}
            if link_id != -1:
                joint_info = get_joint_info(object_id, link_id)
                joint_info['pos'] = p.getJointState(object_id, link_id)[0]
                interaction_post['joint'] = joint_info
            curr['interaction_pre']=interaction_pre 
            curr['interaction_post']=interaction_post
            recorded_data.append(curr)

        # save episode
        episode_dir = os.path.join(save_dir, '{:04d}'.format(sample_i))
        os.makedirs(episode_dir, exist_ok=True)
        for i, step in enumerate(recorded_data):
            save_images(step.pop('imgs_pre'), episode_dir, i, 'step')
            if i == len(recorded_data) - 1:
                save_images(step.pop('imgs_post'), episode_dir, i+1, 'step')
            else:
                step.pop('imgs_post')
            with open(os.path.join(episode_dir, 
                      'info_{:04d}.json'.format(i)), 'w') as fp:
                json.dump(step,fp)
    s.disconnect()

def save_images(imgs, dir, id, prefix):
    if imgs is not None:
        for k,v in imgs.items():
            if k == '3d':
                v = Image.fromarray(v,'I;16')
            else:
                v = Image.fromarray(v)
            v.save(os.path.join(
                   dir, '{}_{:04d}_{}.png'.format(prefix, id, k)))

def get_joint_info(object_id, link_id):
    if link_id == -1:
        return None
    full_info = p.getJointInfo(object_id, link_id)
    return {'type':full_info[2],
            'limit':tuple(full_info[8:10]),
            'axis':tuple(full_info[-4]),
            'parent_pos':tuple(full_info[-3]),
            'parent_orn':tuple(full_info[-2]),
            'parent_idx':full_info[-1]}

def get_link_pose(object_id, link_id):
    link_pos, link_orn = None, None
    if link_id == -1:
        link_pos, link_orn = p.getBasePositionAndOrientation(object_id)
    else:
        link_state = p.getLinkState(object_id, link_id)
        link_pos, link_orn = link_state[:2]
    return tuple(link_pos), tuple(link_orn)


if __name__ == '__main__':
    main()
