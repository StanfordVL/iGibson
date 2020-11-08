'''
Credit: Andrey Kurenkov 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path
import sys
import math
import random
import pickle
import time
import yaml
import copy
import cv2

import numpy as np
import pybullet as pb
import pybullet_data
import matplotlib
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from gibson2.simulator import Simulator
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, InstanceGroup, Instance, quat2rotmat, xyz2mat
from gibson2.objects.articulated_object import ArticulatedObject

CAMERA_UP_VECTOR =[0, 0, 1]

class StaticObject(ArticulatedObject):

    def __init__(self, obj_fname, scale=1.0):
        super(StaticObject, self).__init__(obj_fname, scale)

    def load(self):
        self.body_id = self._load()
        return self.body_id

    def _load(self):
        body_id = pb.loadURDF(self.filename,
                             globalScaling=self.scale,
                             useFixedBase=1,
                             flags=pb.URDF_USE_MATERIAL_COLORS_FROM_MTL)

        return body_id

class ShelfObject(ArticulatedObject):

    def __init__(self, obj_fname, scale=1.0):
        super(ShelfObject, self).__init__(obj_fname, scale)
        is_target_object = False
        urdf_dir = os.path.dirname(obj_fname)
        orientations_path = os.path.join(urdf_dir,'vhacd', 'orientations.yaml')
        if os.path.isfile(orientations_path):
            with open(orientations_path, 'r') as f:
                orientations_dicts = yaml.load(f)
            self.orientations = []
            self.orientation_probs = []
            self.orientation_sizes = []
            total_orientation_prob = 0.0 # Just in case do not sum to 1.0
            for i in range(len(orientations_dicts)):
                total_orientation_prob+=orientations_dicts[i]['prob']
                self.orientations.append(orientations_dicts[i]['rotation'])
                self.orientation_sizes.append(orientations_dicts[i]['size'])
            for i in range(len(orientations_dicts)):
                self.orientation_probs.append(orientations_dicts[i]['prob']/total_orientation_prob)
        else:
            self.orientations = None

    def sample_orientation(self, Z_rotation_randomization = 0):
        if self.orientations is None:
            raise ValueError('No orientations found!')
        orientation = self.orientations[np.random.choice(len(self.orientations),
                                                         p=self.orientation_probs)]
        if Z_rotation_randomization == 0:
            return orientation
        rand_amount = np.random.uniform(-Z_rotation_randomization , Z_rotation_randomization)
        current = Quaternion(array=np.array(orientation))
        rot = Quaternion(axis=(0.0, 0.0, 1.0), radians=rand_amount*math.pi/2)
        new = rot*current
        return [new.real, new.vector[0], new.vector[1], new.vector[2]]

class ObjectContainer(StaticObject):

    def __init__(self, obj_fname, scale=1.0):   # scale=15.0):
        super(StaticObject, self).__init__(obj_fname, scale)
        urdf_dir = os.path.dirname(obj_fname)
        info_file = os.path.join(urdf_dir, 'info.yaml')

        if os.path.isfile(info_file):
            with open(info_file,'r') as f:
                container_info = yaml.load(f)

                self.size = container_info['size']
                self.aabb = container_info['aabb']

                if 'shelf_heights' in container_info:
                    self.shelf_heights = container_info['shelf_heights'] 
                    # self.shelf_heights = [height * 15.0 for height in self.shelf_heights]
        else:
            self.size = [0,0,0]

    def sample_object_set(self, path):
        return objects

class suppress_stdout_stderr(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.
       This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through).

    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fds =  [os.open(os.devnull,os.O_RDWR) for x in range(2)]
        # Save the actual stdout (1) and stderr (2) file descriptors.
        self.save_fds = [os.dup(1), os.dup(2)]

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fds[0],1)
        os.dup2(self.null_fds[1],2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fds[0],1)
        os.dup2(self.save_fds[1],2)
        # Close all file descriptors
        for fd in self.null_fds + self.save_fds:
            os.close(fd)

class ContainerObjectsEnv(object):

    def __init__(self,
            path_to_cfg='cfg/container_env.yaml',
            show_gui=False):
        config = {}
        with open(os.path.join(os.getcwd(), path_to_cfg)) as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # set up our initial configurations
        self.config = config
        self.image_height = config['image_height']
        self.image_width = config['image_width']
        self.camera_height = config.get('camera_height',0.91)
        self.camera_x = config.get('camera_x', 0.65)
        self.camera_y = config.get('camera_y', 0.0)

        self.objects = {}

        if show_gui:
            self.simulator = Simulator(image_width=self.image_width, image_height=self.image_height)
        else:
            self.simulator = Simulator(image_width=self.image_width, image_height=self.image_height,
                                       mode='headless')
                                       # timestep=0.001)
        #self.simulator.use_pb_renderer=True
        pb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pb.loadURDF('plane.urdf')

        # set up renderer
        self.adjust_camera([self.camera_x, self.camera_y, self.camera_height], [0,0,0], [-1,0,0])
        self.simulator.renderer.set_light_pos([0.65, 0.0, 10.0])
        self.simulator.renderer.set_fov(53)

    def reset(self, container):
        with suppress_stdout_stderr():
            self.simulator.reload()
            self.container = container
            self.simulator.import_object(container)
            container.set_position([0,0,container.size[2]/2.0])
            self.objects = {}

    def adjust_camera(self, camera_eye_position, camera_target_position, camera_up_vector, randomize=False):
        if randomize:
            self.simulator.renderer.set_camera(camera_eye_position+np.random.random(3)*0.1,
                                               camera_target_position+np.random.random(3)*0.1,
                                               camera_up_vector)
        else:
            self.simulator.renderer.set_camera(camera_eye_position,
                                               camera_target_position,
                                               camera_up_vector)

    def __del__(self):
        self.simulator.renderer.release()
        del self.simulator.renderer

    def get_renderer_rgb_depth(self):
        rgb, im3d = self.simulator.renderer.render(modes=('rgb','3d'))
        depth = im3d[:,:,2]

        return rgb, -depth

    def get_renderer_rgb(self):
        rgb = self.simulator.renderer.render(modes=('rgb'))[0]
        return rgb

    def get_renderer_depth(self):
        rgb, im3d = self.simulator.renderer.render(modes=('rgb','3d'))
        depth = im3d[:,:,2]

        return -depth

    # selected_object should be an object id
    def get_renderer_segmask(self, selected_object_id, with_occlusion=True):
        all_instances = self.simulator.renderer.instances.copy()
        # get unoccluded segmask
        self.simulator.renderer.instances = []
        selected_object = self.get_body(selected_object_id)

        for instance in all_instances:
            if selected_object is not None and isinstance(instance, Instance) and instance.pybullet_uuid == selected_object:
                self.simulator.renderer.instances.append(instance)
            elif isinstance(instance, Instance) and instance.pybullet_uuid == selected_object_id:
                self.simulator.renderer.instances.append(instance)
        target_segmask = np.sum(self.simulator.renderer.render(modes=('seg'))[0][:,:,:3], axis=2)

        target_segmask, target_depth = self.simulator.renderer.render(modes=('seg', '3d'))
        target_segmask = np.sum(target_segmask[:,:,:3],axis=2)
        target_depth = target_depth[:,:,2]

        self.simulator.renderer.instances=all_instances
        if not with_occlusion:
            return target_segmask

        all_depth = self.simulator.renderer.render(modes=('3d'))[0][:,:,2]
        occluded_segmask = np.logical_and(np.abs(target_depth-all_depth)<0.01, target_segmask)

        return occluded_segmask

    def set_camera_point_at(self, position, randomize=False):
        camera_eye_position = np.asarray([position[0], position[1]-0.3, position[2]+0.1])
        camera_target_position = np.asarray([position[0], position[1], position[2]])
        self.adjust_camera(camera_eye_position,
                           camera_target_position,
                           CAMERA_UP_VECTOR,
                           randomize=randomize)
        self.simulator.sync()

    def get_observation(self,
                        segmak_object_id=None,
                        visualize=False,
                        save=False,
                        demo=False,
                        randomize_camera=False):
        rgb_im, depth_im = self.get_renderer_rgb_depth()
        if segmak_object_id:
            segmask_im = self.get_renderer_segmask(segmak_object_id)

        if visualize:
            plt.imshow(segmask_im)
            plt.title('segmask')
            plt.show()
            plt.close()

            plt.imshow(rgb_im)
            plt.title('rgb')
            plt.show()
            plt.close()

            plt.imshow(depth_im)
            plt.title('depth')
            plt.show()
            plt.close()

        if save:
            save_dir = './rendered_images/'
            save_time = str(time.time())
            plt.imshow(segmask_im)
            plt.title('segmask')
            plt.savefig(save_dir + "segmask_im_" + save_time + '_.png')
            plt.close()

            plt.imshow(rgb_im)
            plt.title('rgb')
            plt.savefig(save_dir + "rgb_im_" + save_time + '_.png')
            plt.close()

            plt.imshow(depth_im)
            plt.title('depth')
            plt.savefig(save_dir + "depth_im_" + save_time + '_.png')
            plt.close()

        if segmak_object_id:
            return rgb_im, depth_im, segmask_im
        return rgb_im, depth_im

    def get_obj_ids(self):
        return list(self.objects.keys())

    def remove_object(self, obj):
        obj_id = obj.body_id
        pb.removeBody(bodyUniqueId=obj_id)
        del self.objects[obj_id]
        self.simulator.sync()

    def add_object(self, obj):
        with suppress_stdout_stderr():
            new_obj_id = self.simulator.import_object(obj)
            self.objects[new_obj_id] = obj
            return new_obj_id

    def get_object(self, body_id):
        return self.get_body(body_id)

    def get_body(self, body_id):
        if body_id in self.objects:
            return self.objects[body_id]
        return None

    def gentle_drop(self, body_id, threshold = 0.1):
        start_time = time.time()

        lvel = pb.getBaseVelocity(bodyUniqueId=body_id)[0]
        avel = pb.getBaseVelocity(bodyUniqueId=body_id)[1]
        pos, _ = pb.getBasePositionAndOrientation(body_id)
        while np.linalg.norm(lvel) < threshold:
            pb.stepSimulation()
            lvel = pb.getBaseVelocity(bodyUniqueId=body_id)[0]
            avel = pb.getBaseVelocity(bodyUniqueId=body_id)[1]
        for _ in range(100000):
            if time.time() > start_time + 5:
                return False
            pos, _ = pb.getBasePositionAndOrientation(body_id)
            if pos[2] < 0:
                return False
            for _ in range(10):
                pb.stepSimulation()
            lvel = pb.getBaseVelocity(bodyUniqueId=body_id)[0]
            avel = pb.getBaseVelocity(bodyUniqueId=body_id)[1]
            # return if it's basically stable
            if np.linalg.norm(lvel) < threshold*0.5 and np.linalg.norm(avel) < threshold:
                return True
            # modify linear velocity if too large, else leave the same
            if np.linalg.norm(lvel) > threshold:
                new_lvel = np.array(lvel) * (threshold / np.linalg.norm(lvel))
            else:
                new_lvel = lvel
            # modify angular velocity if too large, else leave the same
            if np.linalg.norm(avel) > threshold:
                new_avel = np.array(avel) * (threshold / np.linalg.norm(avel))
            else:
                new_avel = avel
            pb.resetBaseVelocity(
                    objectUniqueId=body_id,
                    angularVelocity=list(new_avel),
                    linearVelocity=list(new_lvel))

        return True

    def load_container_setup(self, obj_file, container_id=None):
        abs_path = os.path.join(os.getcwd() + '/', obj_file)

        with open(abs_path, 'rb') as file:
            data = pickle.load(file)

        self.add_objects(data, container_id)

    def get_obj_img(self, color_obs, segmask, save=False):
        np_any = np.any(segmask)
        if not np_any:
            return None
        rows = np.any(segmask, axis=0)
        cols = np.any(segmask, axis=1)
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmean = (rmin+rmax)/2
        cmean = (cmin+cmax)/2
        rlen = rmax - rmin
        clen = cmax - cmin
        maxlen = max([rlen, clen])*1.2
        rmin = int((rmean - maxlen/2))
        rmax = int((rmean + maxlen/2))
        cmin = int((cmean - maxlen/2))
        cmax = int((cmean + maxlen/2))
        obj_color = color_obs[cmin:cmax,rmin:rmax]
        try:
            if save:
                save_dir = './sim_images/'
                plt.imshow(obj_color)
                save_time = str(time.time())
                plt.savefig(save_dir + "im_" + save_time + '_.png')
        except:
            pass
        return obj_color
