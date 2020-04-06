from gibson2.core.physics.scene import StadiumScene
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, InstanceGroup, Instance, quat2rotmat,\
    xyz2mat, xyzw2wxyz
from gibson2.core.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, RBOObject, Pedestrian, ShapeNetObject, BoxShape
from gibson2.core.render.viewer import Viewer
import pybullet as p
import gibson2
import os
import numpy as np


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
                 render_to_tensor=False):
        """
        Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
        both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.

        :param gravity: gravity on z direction.
        :param timestep: timestep of physical simulation
        :param use_fisheye: use fisheye
        :param mode: choose mode from gui or headless
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        :param render_to_tensor: Render to GPU tensors
        """
        # physics simulator
        self.gravity = gravity
        self.timestep = timestep
        self.mode = mode

        # renderer
        self.image_width = image_width
        self.image_height = image_height
        self.vertical_fov = vertical_fov
        self.device_idx = device_idx
        self.use_fisheye = use_fisheye
        self.render_to_tensor = render_to_tensor
        self.load()

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

        if self.mode == 'gui':
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)

        if self.mode == 'gui' and not self.render_to_tensor:
            self.add_viewer()

        self.visual_objects = {}
        self.robots = []
        self.scene = None
        self.objects = []
        self.next_class_id = 0

    def load_without_pybullet_vis(load_func):
        def wrapped_load_func(*args, **kwargs):
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
            res = load_func(*args, **kwargs)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            return res
        return wrapped_load_func

    @load_without_pybullet_vis
    def import_scene(self, scene, texture_scale=1.0, load_texture=True, class_id=None):
        """
        Import a scene. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: Scene object
        :param texture_scale: Option to scale down the texture for rendering
        :param load_texture: If you don't need rgb output, texture loading could be skipped to make rendering faster
        :param class_id: Class id for rendering semantic segmentation
        """

        if class_id is None:
            class_id = self.next_class_id
        self.next_class_id += 1

        new_objects = scene.load()
        for item in new_objects:
            self.objects.append(item)

        for new_object in new_objects:
            for shape in p.getVisualShapeData(new_object):
                id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
                if type == p.GEOM_MESH:
                    filename = filename.decode('utf-8')
                    if not filename in self.visual_objects.keys():
                        self.renderer.load_object(filename,
                                                  texture_scale=texture_scale,
                                                  load_texture=load_texture)
                        self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                        self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                                   pybullet_uuid=new_object,
                                                   class_id=class_id)

                    else:
                        self.renderer.add_instance(self.visual_objects[filename],
                                                   pybullet_uuid=new_object,
                                                   class_id=class_id)
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
                    # self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                    #                            pybullet_uuid=new_object,
                    #                            class_id=class_id)

        if scene.is_interactive:
            for obj in scene.scene_objects:
                class_id = self.next_class_id
                self.next_class_id += 1

                # obj.load() should have already been called in scene.load()
                new_object = obj.body_id

                is_soft = False
                if obj.__class__.__name__ == 'SoftObject':
                    is_soft = True
                self.objects.append(new_object)
                for shape in p.getVisualShapeData(new_object):
                    id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
                    if type == p.GEOM_MESH:
                        filename = filename.decode('utf-8')
                        # print(filename, self.visual_objects)
                        self.renderer.load_object(filename)
                        self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                        self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                                   pybullet_uuid=new_object,
                                                   class_id=class_id,
                                                   dynamic=True,
                                                   softbody=is_soft)

        self.scene = scene
        return new_objects

    @load_without_pybullet_vis
    def import_object(self, object, class_id=None):
        """
        :param object: Object to load
        :param class_id: Class id for rendering semantic segmentation
        """

        if class_id is None:
            class_id = self.next_class_id
        self.next_class_id += 1

        new_object = object.load()
        is_soft = False
        if object.__class__.__name__ == 'SoftObject':
            is_soft = True
        self.objects.append(new_object)
        for shape in p.getVisualShapeData(new_object):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                # print(filename, self.visual_objects)
                if not filename in self.visual_objects.keys():
                    self.renderer.load_object(filename)
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                    self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                               pybullet_uuid=new_object,
                                               class_id=class_id,
                                               dynamic=True,
                                               softbody=is_soft)
                else:
                    self.renderer.add_instance(self.visual_objects[filename],
                                               pybullet_uuid=new_object,
                                               class_id=class_id,
                                               dynamic=True,
                                               softbody=is_soft)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                self.renderer.load_object(
                    filename,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                           pybullet_uuid=new_object,
                                           class_id=class_id,
                                           dynamic=True)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                           pybullet_uuid=new_object,
                                           class_id=class_id,
                                           dynamic=True)
            elif type == p.GEOM_BOX:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=[dimensions[0], dimensions[1], dimensions[2]])
                self.renderer.add_instance(len(self.renderer.visual_objects) - 1,
                                           pybullet_uuid=new_object,
                                           class_id=class_id,
                                           dynamic=True)

        return new_object

    @load_without_pybullet_vis
    def import_robot(self, robot, class_id=None):
        """
        Import a robot into Simulator

        :param robot: Robot
        :param class_id: Class id for rendering semantic segmentation
        :return: id for robot in pybullet
        """

        if class_id is None:
            class_id = self.next_class_id
        self.next_class_id += 1

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
                if not filename in self.visual_objects.keys():
                    # print(filename, rel_pos, rel_orn, color, dimensions)
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))

                    visual_objects.append(len(self.renderer.visual_objects) - 1)
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                else:
                    visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                # print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                # print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                # print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=[dimensions[0], dimensions[1], dimensions[2]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        self.renderer.add_robot(object_ids=visual_objects,
                                link_ids=link_ids,
                                pybullet_uuid=ids[0],
                                class_id=class_id,
                                poses_rot=poses_rot,
                                poses_trans=poses_trans,
                                dynamic=True,
                                robot=robot)

        return ids

    @load_without_pybullet_vis
    def import_articulated_object(self, obj, class_id=None):
        """
        Import articulated objects into simulator

        :param obj:
        :param class_id: Class id for rendering semantic segmentation
        :return: pybulet id
        """

        if class_id is None:
            class_id = self.next_class_id
        self.next_class_id += 1

        ids = obj.load()
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []

        for shape in p.getVisualShapeData(ids):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if not filename in self.visual_objects.keys():
                    # print(filename, rel_pos, rel_orn, color, dimensions)
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))

                    visual_objects.append(len(self.renderer.visual_objects) - 1)
                    self.visual_objects[filename] = len(self.renderer.visual_objects) - 1
                else:
                    visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                # print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                # print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                # print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=[dimensions[0], dimensions[1], dimensions[2]])
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)

            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        self.renderer.add_instance_group(object_ids=visual_objects,
                                         link_ids=link_ids,
                                         pybullet_uuid=ids,
                                         class_id=class_id,
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
        self.sync()

    def sync(self):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.update_position(instance)
        if self.mode == 'gui' and self.viewer is not None:
            self.viewer.update()

    @staticmethod
    def update_position(instance):
        """
        Update position for an object or a robot in renderer.

        :param instance: Instance in the renderer
        """
        if isinstance(instance, Instance):
            pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid) # orn is in x,y,z,w
            instance.set_position(pos)
            instance.set_rotation(xyzw2wxyz(orn))
        elif isinstance(instance, InstanceGroup):
            poses_rot = []
            poses_trans = []

            for link_id in instance.link_ids:
                if link_id == -1:
                    pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
                else:
                    _, _, _, _, pos, orn = p.getLinkState(instance.pybullet_uuid, link_id)
                poses_rot.append(
                    np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
                poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))
                # print(instance.pybullet_uuid, link_id, pos, orn)

            instance.poses_rot = poses_rot
            instance.poses_trans = poses_trans

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
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)
        self.renderer.release()
