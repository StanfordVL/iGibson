from gibson2.core.physics.scene import StadiumScene
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, InstanceGroup, Instance, quat2rotmat, xyz2mat, xyzw2wxyz
from gibson2.core.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, RBOObject, Pedestrian, ShapeNetObject, BoxShape
from gibson2.core.render.viewer import Viewer
from gibson2.utils.assets_utils import get_model_path
import pybullet as p
import gibson2
import os
import numpy as np
import collections
import cv2

import platform
import logging


class Simulator:
    def __init__(self,
                 gravity=9.8,
                 timestep=1 / 240.0,
                 use_fisheye=False,
                 mode='gui',
                 image_width=128,
                 image_height=128,
                 vertical_fov=90,
                 device_idx=0,
                 render_to_tensor=False,
                 auto_sync=True):
        """
        Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
        both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.

        :param gravity: gravity on z direction.
        :param timestep: timestep of physical simulation
        :param use_fisheye: use fisheye
        :param mode: choose mode from gui, headless, iggui (only open iGibson UI), or pbgui(only open pybullet UI)
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        :param render_to_tensor: Render to GPU tensors
        :param auto_sync: automatically sync object poses to gibson renderer, by default true,
        disable it when you want to run multiple physics step but don't need to visualize each frame
        """
        # physics simulator
        self.gravity = gravity
        self.timestep = timestep
        self.mode = mode

        plt = platform.system()
        if plt == 'Darwin' and self.mode == 'gui':
            self.mode = 'iggui' # for mac os disable pybullet rendering
            logging.warn('Rendering both iggui and pbgui is not supported on mac, choose either pbgui or '
                      'iggui. Default to iggui.')

        self.use_pb_renderer = False
        self.use_ig_renderer = False

        if self.mode in ['gui', 'iggui']:
            self.use_ig_renderer = True

        if self.mode in ['gui', 'pbgui']:
            self.use_pb_renderer = True

        # renderer
        self.image_width = image_width
        self.image_height = image_height
        self.vertical_fov = vertical_fov
        self.device_idx = device_idx
        self.use_fisheye = use_fisheye
        self.render_to_tensor = render_to_tensor
        self.auto_sync = auto_sync
        self.load()

        # keeps track of semantic classes and instance counts
        self.class_instance_tracker = collections.Counter()
        self.class_instance_tracker_scene_only = collections.Counter()
        # valid semantic class id starts from 1
        # semantic class id 0 is reserved for background
        self.next_class_id = 1

    def set_timestep(self, timestep):
        """
        :param timestep: set timestep after the initialization of Simulator
        """
        self.timestep = timestep
        p.setTimeStep(self.timestep)

    def add_viewer(self):
        """
        Attach a debugging viewer to the renderer. This will make the step much slower so should be avoided when
        training agents
        """
        self.viewer = Viewer()
        self.viewer.renderer = self.renderer

    def reload(self):
        """
        Destroy the MeshRenderer and physics simulator and start again.
        """
        self.disconnect()
        self.load()

    def load(self):
        """
        Set up MeshRenderer and physics simulation client. Initialize the list of objects.
        """
        if self.render_to_tensor:
            self.renderer = MeshRendererG2G(width=self.image_width,
                                            height=self.image_height,
                                            vertical_fov=self.vertical_fov,
                                            device_idx=self.device_idx,
                                            use_fisheye=self.use_fisheye)
        else:
            self.renderer = MeshRenderer(width=self.image_width,
                                         height=self.image_height,
                                         vertical_fov=self.vertical_fov,
                                         device_idx=self.device_idx,
                                         use_fisheye=self.use_fisheye)

        print("******************PyBullet Logging Information:")
        if self.use_pb_renderer:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        print("PyBullet Logging Information******************")

        if self.use_ig_renderer and not self.render_to_tensor:
            self.add_viewer()

        self.visual_objects = {}
        self.robots = []
        self.scene = None
        self.objects = []

    def load_without_pybullet_vis(load_func):
        def wrapped_load_func(*args, **kwargs):
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
            res = load_func(*args, **kwargs)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            return res
        return wrapped_load_func

    def update_class_instance_tracker(self, class_id, instance_count, is_scene=False):
        if is_scene:
            if class_id in self.class_instance_tracker:
                print('WARNING: scene class id [%d] already exists in self.class_instance_tracker. '
                    'One possible solution is to import_scene first before import_robot or import_object.' % class_id)
        else:
            if class_id in self.class_instance_tracker_scene_only:
                print('WARNING: object class id [%d] conflicts with scene class id. '
                    'One possible solution is to add object with a different class_id that does not exist in the scene.' % class_id)

        self.class_instance_tracker[class_id] += instance_count

        return self.class_instance_tracker[class_id]

    def get_next_available_class_id(self):
        while self.next_class_id in self.class_instance_tracker:
            self.next_class_id += 1
        if not (1 <= self.next_class_id <= 4095):
            raise Exception('currently class id can only from 1 to 4095 (inclusive), 0 is reserved for background.')
        return self.next_class_id

    @load_without_pybullet_vis
    def import_scene(self, scene, texture_scale=1.0, load_texture=True, load_sem_map=True, class_id=None):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: Scene object
        :param texture_scale: Option to scale down the texture for rendering
        :param load_texture: If you don't need rgb output, texture loading could be skipped to make rendering faster
        :param load_sem_map: Whether to load semantic class and instance maps of the scene if they exist
        :param class_id: Class id for rendering semantic segmentation
        """

        if class_id is None:
            class_id = self.get_next_available_class_id()

        new_objects = scene.load()
        for item in new_objects:
            self.objects.append(item)

        for new_object in new_objects:
            for shape in p.getVisualShapeData(new_object):
                id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
                visual_object = None
                if type == p.GEOM_MESH:
                    filename = filename.decode('utf-8')
                    if filename not in self.visual_objects.keys():
                        self.renderer.load_object(filename,
                                                  texture_scale=texture_scale,
                                                  load_texture=load_texture,
                                                  load_sem_map=load_sem_map)
                        self.visual_objects[filename] = len(
                            self.renderer.visual_objects) - 1
                    visual_object = self.visual_objects[filename]
                elif type == p.GEOM_PLANE:
                    pass
                    # By default, we add an additional floor surface to "smooth out" that of the original mesh.
                    # Normally you don't need to render this additionally added floor surface.
                    # However, if you do want to render it for some reason, you can uncomment the block below.

                    # filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                    # self.renderer.load_object(filename,
                    #                           transform_orn=rel_orn,
                    #                           transform_pos=rel_pos,
                    #                           input_kd=color[:3],
                    #                           scale=[100, 100, 0.01])
                    # visual_object = len(self.renderer.visual_objects) - 1

                if visual_object is not None:
                    self.renderer.add_instance(visual_object,
                                               pybullet_uuid=new_object,
                                               class_id=class_id,
                                               instance_id=0)


        # if load_sem_map is True and the house_metadata_file can be found,
        # sem_map.png and ins_map.png will be used for rendering semantic and instance segmentation.
        if load_sem_map and scene.model_id is not None:
            # matterport house metadata file
            house_metadata_file = os.path.join(get_model_path(scene.model_id), 'house_segmentations', '{}.house'.format(scene.model_id))
            if os.path.isfile(house_metadata_file):
                self.class_instance_tracker_scene_only = collections.Counter()
                with open(house_metadata_file) as f:
                    for line in f.readlines():
                        if line.startswith('O'):
                            ls = line.strip().split()
                            # category id is actually 0-indexed, +1 to be consistent with https://github.com/niessner/Matterport/blob/master/metadata/category_mapping.tsv
                            category_id = int(ls[3]) + 1
                            self.class_instance_tracker_scene_only[category_id] += 1

        # otherwise, the provided class_id and instance_id = 0 will be used
        else:
            self.class_instance_tracker_scene_only = collections.Counter([class_id])

        for class_id in self.class_instance_tracker_scene_only:
            self.update_class_instance_tracker(class_id, self.class_instance_tracker_scene_only[class_id], is_scene=True)

        scene_object_ids = []
        if scene.is_interactive:
            # Currently treating each scene object as a separate class
            # TODO: scene objects should have semantic class information
            for obj in scene.scene_objects:
                self.import_articulated_object(obj)
                scene_object_ids.append(obj.body_id)

        self.scene = scene

        return new_objects + scene_object_ids

    @load_without_pybullet_vis
    def import_object(self, obj, class_id=None):
        """
        Import a non-articulated object into the simulator

        :param obj: Object to load
        :param class_id: Class id for rendering semantic segmentation
        """

        if class_id is None:
            class_id = self.get_next_available_class_id()

        new_object = obj.load()
        softbody = False
        if obj.__class__.__name__ == 'SoftObject':
            softbody = True

        self.objects.append(new_object)

        for shape in p.getVisualShapeData(new_object):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            visual_object = None
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if filename not in self.visual_objects.keys():
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))
                    self.visual_objects[filename] = len(
                        self.renderer.visual_objects) - 1
                visual_object = self.visual_objects[filename]
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_object = len(self.renderer.visual_objects) - 1
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_object = len(self.renderer.visual_objects) - 1
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_object = len(self.renderer.visual_objects) - 1

            if visual_object is not None:
                # instance_id is 0-indexed
                instance_id = self.update_class_instance_tracker(class_id, 1) - 1
                self.renderer.add_instance(visual_object,
                                           pybullet_uuid=new_object,
                                           class_id=class_id,
                                           instance_id=instance_id,
                                           dynamic=True,
                                           softbody=softbody)
        return new_object

    @load_without_pybullet_vis
    def import_robot(self, robot, class_id=None):
        """
        Import a robot into the simulator

        :param robot: Robot
        :param class_id: Class id for rendering semantic segmentation
        :return: pybullet id
        """

        if class_id is None:
            class_id = self.get_next_available_class_id()

        ids = robot.load()
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []
        self.robots.append(robot)

        for shape in p.getVisualShapeData(ids[0]):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if filename not in self.visual_objects.keys():
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))
                    self.visual_objects[filename] = len(
                        self.renderer.visual_objects) - 1
                visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)

            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        # instance_id is 0-indexed
        instance_id = self.update_class_instance_tracker(class_id, 1) - 1
        self.renderer.add_robot(object_ids=visual_objects,
                                pybullet_link_ids=link_ids,
                                pybullet_uuid=ids[0],
                                class_id=class_id,
                                instance_id=instance_id,
                                poses_rot=poses_rot,
                                poses_trans=poses_trans,
                                dynamic=True,
                                robot=robot)

        return ids

    @load_without_pybullet_vis
    def import_articulated_object(self, obj, class_id=None, material_override=None):
        """
        Import an articulated object into the simulator

        :param obj: Object to load
        :param class_id: Class id for rendering semantic segmentation
        :return: pybulet id
        """

        if class_id is None:
            class_id = self.get_next_available_class_id()

        ids = obj.load()
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []

        for shape in p.getVisualShapeData(ids):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if filename not in self.visual_objects.keys():
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions),
                                              material_override=material_override)
                    self.visual_objects[filename] = len(
                        self.renderer.visual_objects) - 1
                visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)

            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        instance_id = self.update_class_instance_tracker(class_id, 1)
        self.renderer.add_instance_group(object_ids=visual_objects,
                                         pybullet_link_ids=link_ids,
                                         pybullet_uuid=ids,
                                         class_id=class_id,
                                         instance_id=instance_id,
                                         poses_rot=poses_rot,
                                         poses_trans=poses_trans,
                                         dynamic=True,
                                         robot=None)

        return ids

    def step(self):
        """
        Step the simulation and update positions in renderer
        """

        p.stepSimulation()
        if self.auto_sync:
            self.sync()

    def sync(self):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.update_position(instance)
        if self.use_ig_renderer and self.viewer is not None:
            self.viewer.update()

    @staticmethod
    def update_position(instance):
        """
        Update position for an object or a robot in renderer.

        :param instance: Instance in the renderer
        """
        if isinstance(instance, Instance):
            pos, orn = p.getBasePositionAndOrientation(
                instance.pybullet_uuid)  # orn is in x,y,z,w
            instance.set_position(pos)
            instance.set_rotation(xyzw2wxyz(orn))
        elif isinstance(instance, InstanceGroup):
            poses_rot = []
            poses_trans = []

            for link_id in instance.pybullet_link_ids:
                if link_id == -1:
                    pos, orn = p.getBasePositionAndOrientation(
                        instance.pybullet_uuid)
                else:
                    _, _, _, _, pos, orn = p.getLinkState(
                        instance.pybullet_uuid, link_id)

                poses_rot.append(np.ascontiguousarray(
                    quat2rotmat(xyzw2wxyz(orn))))
                poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

            #instance.poses_rot = poses_rot
            #instance.poses_trans = poses_trans
            instance.set_rotation(poses_rot)
            instance.set_position(poses_trans)
    def isconnected(self):
        """
        :return: pybullet is alive
        """
        return p.getConnectionInfo(self.cid)['isConnected']

    def disconnect(self):
        """
        clean up the simulator
        """
        if self.isconnected():
            print("******************PyBullet Logging Information:")
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)
            print("PyBullet Logging Information******************")
        self.renderer.release()
