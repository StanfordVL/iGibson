from gibson2.core.physics.scene import StadiumScene
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, InstanceGroup, Instance, quat2rotmat, xyz2mat
#from gibson2.core.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.core.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR
from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject, RBOObject, Pedestrian, ShapeNetObject, BoxShape
from gibson2.core.render.viewer import Viewer, ViewerVR
import pybullet as p
import gibson2
import os
import numpy as np
import time

# Note: pass in mode='vr' to use the simulator in VR mode
# Note 2: vrWidth and vrHeight can be set to manually change the VR resolution
# It is, however, recommended to use the VR headset's native 2016 x 2240 resolution where possible
class Simulator:
    def __init__(self,
                 gravity=9.8,
                 timestep=1 / 240.0,
                 use_fisheye=False,
                 mode='gui',
                 resolution=256,
                 fov=90,
                 device_idx=0,
                 render_to_tensor=False,
                 vrWidth=None,
                 vrHeight=None):
        """
        Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
        both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.

        :param gravity: gravity on z direction.
        :param timestep: timestep of physical simulation
        :param use_fisheye: use fisheye
        :param mode: choose mode from gui or headless
        :param resolution: resolution of camera (square)
        :param fov: field of view of camera in degree
        :param device_idx: GPU device index to run rendering on
        :param render_to_tensor: Render to GPU tensors
        """
        # physics simulator
        self.gravity = gravity
        self.timestep = timestep
        self.mode = mode
        # renderer
        self.resolution = resolution
        self.vrWidth = vrWidth
        self.vrHeight = vrHeight
        self.fov = fov
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
        if self.mode == 'vr':
            self.viewer = ViewerVR()
        else:
            self.viewer = Viewer()
        self.viewer.renderer = self.renderer

    def reload(self):
        """
        Destroy the MeshRenderer and physics simulator and start again.
        """
        self.renderer.release()
        p.disconnect(self.cid)
        self.load()

    def load(self):

        """
        Set up MeshRenderer and physics simulation client. Initialize the list of objects.
        """
        if self.render_to_tensor:
            self.renderer = MeshRendererG2G(width=self.resolution,
                                         height=self.resolution,
                                         fov=self.fov,
                                         device_idx=self.device_idx,
                                         use_fisheye=self.use_fisheye)
        elif self.mode == 'vr':
            # TODO: Add options to change the mesh renderer that VR renderer takes in here
            self.renderer = MeshRendererVR(MeshRenderer, vrWidth=self.vrWidth, vrHeight=self.vrHeight)
        else:
            self.renderer = MeshRenderer(width=self.resolution,
                                     height=self.resolution,
                                     fov=self.fov,
                                     device_idx=self.device_idx,
                                     use_fisheye=self.use_fisheye)

        # Connect directly to pybullet when using VR
        if self.mode == 'vr':
            self.cid = p.connect(p.DIRECT)
        elif self.mode == 'gui':
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setTimeStep(self.timestep)
        p.setGravity(0, 0, -self.gravity)

        if (self.mode == 'gui' or self.mode == 'vr') and not self.render_to_tensor:
            self.add_viewer()

        self.visual_objects = {}
        self.robots = []
        self.scene = None
        self.objects = []

    def import_scene(self, scene, texture_scale=1.0, load_texture=True, class_id=0):

        """
        Import a scene. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: Scene object
        :param texture_scale: Option to scale down the texture for rendering
        :param load_texture: If you don't need rgb output, texture loading could be skipped to make rendering faster
        :param class_id: The class_id for background for rendering semantics.
        """

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
                        self.visual_objects[filename] = len(self.renderer.get_visual_objects()) - 1
                        self.renderer.add_instance(len(self.renderer.get_visual_objects()) - 1,
                                                   pybullet_uuid=new_object,
                                                   class_id=class_id)

                    else:
                        self.renderer.add_instance(self.visual_objects[filename],
                                                   pybullet_uuid=new_object,
                                                   class_id=class_id)
        

                elif type == p.GEOM_PLANE:
                    pass #don't load plane, it will cause z fighting
                    #filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                    # self.renderer.load_object(filename,
                    #                           transform_orn=rel_orn,
                    #                           transform_pos=rel_pos,
                    #                           input_kd=color[:3],
                    #                           scale=[100, 100, 0.01])
                    # self.renderer.add_instance(len(self.renderer.get_visual_objects()) - 1,
                    #                            pybullet_uuid=new_object,
                    #                            class_id=class_id,
                    #                            dynamic=True)

        self.scene = scene
        return new_objects

    def import_object(self, object, class_id=0):
        """
        :param object: Object to load
        :param class_id: class_id to show for semantic segmentation mask
        """
        new_object = object.load()
        isSoft = False
        if object.__class__.__name__ == 'SoftObject':
            isSoft = True
        self.objects.append(new_object)
        for shape in p.getVisualShapeData(new_object):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                print(filename, self.visual_objects)
                if not filename in self.visual_objects.keys():
                    self.renderer.load_object(filename)
                    self.visual_objects[filename] = len(self.renderer.get_visual_objects()) - 1
                    self.renderer.add_instance(len(self.renderer.get_visual_objects()) - 1,
                                               pybullet_uuid=new_object,
                                               class_id=class_id,
                                               dynamic=True,
                                               softbody=isSoft)
                else:
                    self.renderer.add_instance(self.visual_objects[filename],
                                               pybullet_uuid=new_object,
                                               class_id=class_id,
                                               dynamic=True,
                                               softbody=isSoft)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                self.renderer.load_object(
                    filename,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                self.renderer.add_instance(len(self.renderer.get_visual_objects()) - 1,
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
                self.renderer.add_instance(len(self.renderer.get_visual_objects()) - 1,
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
                self.renderer.add_instance(len(self.renderer.get_visual_objects()) - 1,
                                           pybullet_uuid=new_object,
                                           class_id=class_id,
                                           dynamic=True)

        return new_object

    def import_robot(self, robot, class_id=0):
        """
        Import a robot into Simulator

        :param robot: Robot
        :param class_id: class_id to show for semantic segmentation mask
        :return: id for robot in pybullet
        """

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
                    print(filename, rel_pos, rel_orn, color, dimensions)
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))

                    visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                    self.visual_objects[filename] = len(self.renderer.get_visual_objects()) - 1
                else:
                    visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=[dimensions[0], dimensions[1], dimensions[2]])
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat([orn[-1], orn[0], orn[1], orn[2]])))
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

    def import_interactive_object(self, obj, class_id=0):
        """
        Import articulated objects into simulator

        :param obj:
        :param class_id: class_id to show for semantic segmentation mask
        :return: pybulet id
        """
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
                    print(filename, rel_pos, rel_orn, color, dimensions)
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))

                    visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                    self.visual_objects[filename] = len(self.renderer.get_visual_objects()) - 1
                else:
                    visual_objects.append(self.visual_objects[filename])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                # self.visual_objects[filename] = len(self.renderer.get_visual_objects()) - 1
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                print(filename, dimensions, rel_pos, rel_orn, color)
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=[dimensions[0], dimensions[1], dimensions[2]])
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)

            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat([orn[-1], orn[0], orn[1], orn[2]])))
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

    def step(self, should_measure_fps=False):
        """
        Step the simulation and update positions in renderer
        """
        step_start_time = None
        step_end_time = None
        if (should_measure_fps):
            step_start_time = time.time()
        p.stepSimulation()
        # TODO: Explain change to getInstances function - so it can pass through VR layer
        for instance in self.renderer.get_instances():
            if instance.dynamic:
                self.update_position(instance)
        if (self.mode == 'gui' or self.mode == 'vr') and not self.viewer is None:
            self.viewer.update()
        if (should_measure_fps):
            step_end_time = time.time()
            fps = 1/float(step_end_time - step_start_time)
            print("Current fps: %f" % fps)

    @staticmethod
    def update_position(instance):
        """
        Update position for an object or a robot in renderer.

        :param instance: Instance in the renderer
        """
        if isinstance(instance, Instance):
            pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
            instance.set_position(pos)
            instance.set_rotation([orn[-1], orn[0], orn[1], orn[2]])
        elif isinstance(instance, InstanceGroup):
            poses_rot = []
            poses_trans = []

            for link_id in instance.link_ids:
                if link_id == -1:
                    pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
                else:
                    _, _, _, _, pos, orn = p.getLinkState(instance.pybullet_uuid, link_id)
                poses_rot.append(
                    np.ascontiguousarray(quat2rotmat([orn[-1], orn[0], orn[1], orn[2]])))
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
            p.disconnect(self.cid)
        self.renderer.release()
