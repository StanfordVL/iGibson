from igibson.utils.mesh_util import quat2rotmat, xyzw2wxyz, xyz2mat
from igibson.utils.semantics_utils import get_class_name_to_class_id
from igibson.utils.constants import SemanticClass, PyBulletSleepState
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.instances import InstanceGroup, Instance, Robot
from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.viewer import Viewer
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.scene_base import Scene
from igibson.robots.robot_base import BaseRobot
from igibson.objects.object_base import Object


import pybullet as p
import igibson
import os
import numpy as np
import platform
import logging


class Simulator:
    """
    Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
    both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.
    """

    def __init__(self,
                 gravity=9.8,
                 physics_timestep=1 / 120.0,
                 render_timestep=1 / 30.0,
                 mode='gui',
                 image_width=128,
                 image_height=128,
                 vertical_fov=90,
                 device_idx=0,
                 render_to_tensor=False,
                 rendering_settings=MeshRendererSettings()):
        """
        :param gravity: gravity on z direction.
        :param physics_timestep: timestep of physical simulation, p.stepSimulation()
        :param render_timestep: timestep of rendering, and Simulator.step() function
        :param mode: choose mode from gui, headless, iggui (only open iGibson UI), or pbgui(only open pybullet UI)
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        :param render_to_tensor: Render to GPU tensors
        :param rendering_settings: rendering setting
        """
        # physics simulator
        self.gravity = gravity
        self.physics_timestep = physics_timestep
        self.render_timestep = render_timestep
        self.mode = mode

        # TODO: remove this, currently used for testing only
        self.objects = []

        plt = platform.system()
        if plt == 'Darwin' and self.mode == 'gui':
            self.mode = 'iggui'  # for mac os disable pybullet rendering
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
        self.render_to_tensor = render_to_tensor
        self.optimized_renderer = rendering_settings.optimized
        self.rendering_settings = rendering_settings
        self.viewer = None
        self.load()

        self.class_name_to_class_id = get_class_name_to_class_id()
        self.body_links_awake = 0
        self.first_sync = True

    def set_timestep(self, physics_timestep, render_timestep):
        """
        Set physics timestep and render (action) timestep

        :param physics_timestep: physics timestep for pybullet
        :param render_timestep: rendering timestep for renderer
        """
        self.physics_timestep = physics_timestep
        self.render_timestep = render_timestep
        p.setTimeStep(self.physics_timestep)

    def add_viewer(self):
        """
        Attach a debugging viewer to the renderer.
        This will make the step much slower so should be avoided when training agents
        """
        self.viewer = Viewer(simulator=self, renderer=self.renderer)

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
                                            rendering_settings=self.rendering_settings)
        else:
            self.renderer = MeshRenderer(width=self.image_width,
                                         height=self.image_height,
                                         vertical_fov=self.vertical_fov,
                                         device_idx=self.device_idx,
                                         rendering_settings=self.rendering_settings)

        print("******************PyBullet Logging Information:")
        if self.use_pb_renderer:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setTimeStep(self.physics_timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        print("PyBullet Logging Information******************")

        self.visual_objects = {}
        self.robots = []
        self.scene = None

        if self.use_ig_renderer and not self.render_to_tensor:
            self.add_viewer()

    def load_without_pybullet_vis(load_func):
        """
        Load without pybullet visualizer
        """
        def wrapped_load_func(*args, **kwargs):
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, False)
            res = load_func(*args, **kwargs)
            p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, True)
            return res
        return wrapped_load_func

    @load_without_pybullet_vis
    def import_scene(self,
                     scene,
                     texture_scale=1.0,
                     load_texture=True,
                     render_floor_plane=False,
                     class_id=SemanticClass.SCENE_OBJS,
                     ):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: Scene object
        :param texture_scale: Option to scale down the texture for rendering
        :param load_texture: If you don't need rgb output, texture loading could be skipped to make rendering faster
        :param render_floor_plane: Whether to render the additionally added floor plane
        :param class_id: Class id for rendering semantic segmentation
        :return: pybullet body ids from scene.load function
        """
        assert isinstance(scene, Scene) and not isinstance(scene, InteractiveIndoorScene), \
            'import_scene can only be called with Scene that is not InteractiveIndoorScene'
        # Load the scene. Returns a list of pybullet ids of the objects loaded that we can use to
        # load them in the renderer
        new_object_pb_ids = scene.load()
        self.objects += new_object_pb_ids

        # Load the objects in the renderer
        for new_object_pb_id in new_object_pb_ids:
            self.load_object_in_renderer(new_object_pb_id, class_id=class_id, texture_scale=texture_scale,
                                         load_texture=load_texture, render_floor_plane=render_floor_plane,
                                         use_pbr=False, use_pbr_mapping=False)

        self.scene = scene
        return new_object_pb_ids

    @load_without_pybullet_vis
    def import_ig_scene(self, scene):
        """
        Import scene from iGSDF class

        :param scene: iGSDFScene instance
        :return: pybullet body ids from scene.load function
        """
        assert isinstance(scene, InteractiveIndoorScene), \
            'import_ig_scene can only be called with InteractiveIndoorScene'
        new_object_ids = scene.load()
        self.objects += new_object_ids
        if scene.texture_randomization:
            # use randomized texture
            for body_id, visual_mesh_to_material in \
                    zip(new_object_ids, scene.visual_mesh_to_material):
                shadow_caster = True
                if scene.objects_by_id[body_id].category == 'ceilings':
                    shadow_caster = False
                class_id = self.class_name_to_class_id.get(
                    scene.objects_by_id[body_id].category, SemanticClass.SCENE_OBJS)
                self.load_articulated_object_in_renderer(
                    body_id,
                    class_id=class_id,
                    visual_mesh_to_material=visual_mesh_to_material,
                    shadow_caster=shadow_caster)
        else:
            # use default texture
            for body_id in new_object_ids:
                use_pbr = True
                use_pbr_mapping = True
                shadow_caster = True
                if scene.scene_source == 'IG':
                    if scene.objects_by_id[body_id].category in ['walls', 'floors', 'ceilings']:
                        use_pbr = False
                        use_pbr_mapping = False
                if scene.objects_by_id[body_id].category == 'ceilings':
                    shadow_caster = False
                class_id = self.class_name_to_class_id.get(
                    scene.objects_by_id[body_id].category, SemanticClass.SCENE_OBJS)
                self.load_articulated_object_in_renderer(
                    body_id,
                    class_id=class_id,
                    use_pbr=use_pbr,
                    use_pbr_mapping=use_pbr_mapping,
                    shadow_caster=shadow_caster)
        self.scene = scene

        return new_object_ids

    @load_without_pybullet_vis
    def import_object(self,
                      obj,
                      class_id=SemanticClass.USER_ADDED_OBJS,
                      use_pbr=True,
                      use_pbr_mapping=True,
                      shadow_caster=True):
        """
        Import an object into the simulator

        :param obj: Object to load
        :param class_id: Class id for rendering semantic segmentation
        :param use_pbr: Whether to use pbr
        :param use_pbr_mapping: Whether to use pbr mapping
        :param shadow_caster: Whether to cast shadow
        """
        assert isinstance(obj, Object), \
            'import_object can only be called with Object'
        # Load the object in pybullet. Returns a pybullet id that we can use to load it in the renderer
        new_object_pb_id = obj.load()
        self.objects += [new_object_pb_id]
        if obj.__class__ in [ArticulatedObject, URDFObject]:
            self.load_articulated_object_in_renderer(new_object_pb_id,
                                                     class_id,
                                                     use_pbr=use_pbr,
                                                     use_pbr_mapping=use_pbr_mapping,
                                                     shadow_caster=shadow_caster)
        else:
            softbody = False
            if obj.__class__.__name__ == 'SoftObject':
                softbody = True
            self.load_object_in_renderer(new_object_pb_id,
                                         class_id,
                                         softbody,
                                         use_pbr=use_pbr,
                                         use_pbr_mapping=use_pbr_mapping,
                                         shadow_caster=shadow_caster)
        return new_object_pb_id

    @load_without_pybullet_vis
    def load_object_in_renderer(self,
                                object_pb_id,
                                class_id=None,
                                softbody=False,
                                texture_scale=1.0,
                                load_texture=True,
                                render_floor_plane=False,
                                use_pbr=True,
                                use_pbr_mapping=True,
                                shadow_caster=True
                                ):
        """
        Load the object into renderer

        :param object_pb_id: pybullet body id
        :param class_id: Class id for rendering semantic segmentation
        :param softbody: Whether the object is soft body
        :param texture_scale: Texture scale
        :param load_texture: If you don't need rgb output, texture loading could be skipped to make rendering faster
        :param render_floor_plane: Whether to render the additionally added floor plane
        :param use_pbr: Whether to use pbr
        :param use_pbr_mapping: Whether to use pbr mapping
        :param shadow_caster: Whether to cast shadow
        """
        for shape in p.getVisualShapeData(object_pb_id):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            visual_object = None
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if (filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn)) not in self.visual_objects.keys():
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions),
                                              texture_scale=texture_scale,
                                              load_texture=load_texture)
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))
                                        ] = len(self.renderer.visual_objects) - 1
                visual_object = self.visual_objects[
                    (filename,
                     tuple(dimensions),
                     tuple(rel_pos),
                     tuple(rel_orn)
                     )]
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(
                    igibson.assets_path, 'models/mjcf_primitives/sphere8.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_object = len(self.renderer.visual_objects) - 1
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(
                    igibson.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_object = len(self.renderer.visual_objects) - 1
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    igibson.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_object = len(self.renderer.visual_objects) - 1
            elif type == p.GEOM_PLANE:
                # By default, we add an additional floor surface to "smooth out" that of the original mesh.
                # Normally you don't need to render this additionally added floor surface.
                # However, if you do want to render it for some reason, you can set render_floor_plane to be True.
                if render_floor_plane:
                    filename = os.path.join(
                        igibson.assets_path,
                        'models/mjcf_primitives/cube.obj')
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=[100, 100, 0.01])
                    visual_object = len(self.renderer.visual_objects) - 1
            if visual_object is not None:
                self.renderer.add_instance(visual_object,
                                           pybullet_uuid=object_pb_id,
                                           class_id=class_id,
                                           dynamic=True,
                                           softbody=softbody,
                                           use_pbr=use_pbr,
                                           use_pbr_mapping=use_pbr_mapping,
                                           shadow_caster=shadow_caster
                                           )

    @load_without_pybullet_vis
    def load_articulated_object_in_renderer(self,
                                            object_pb_id,
                                            class_id=None,
                                            visual_mesh_to_material=None,
                                            use_pbr=True,
                                            use_pbr_mapping=True,
                                            shadow_caster=True):
        """
        Load the articulated object into renderer

        :param object_pb_id: pybullet body id
        :param class_id: Class id for rendering semantic segmentation
        :param visual_mesh_to_material: mapping from visual mesh to randomizable materials
        :param use_pbr: Whether to use pbr
        :param use_pbr_mapping: Whether to use pbr mapping
        :param shadow_caster: Whether to cast shadow
        """

        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []

        for shape in p.getVisualShapeData(object_pb_id):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            if type == p.GEOM_MESH:
                filename = filename.decode('utf-8')
                if (filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn)) not in self.visual_objects.keys():
                    overwrite_material = None
                    if visual_mesh_to_material is not None and filename in visual_mesh_to_material:
                        overwrite_material = visual_mesh_to_material[filename]
                    self.renderer.load_object(
                        filename,
                        transform_orn=rel_orn,
                        transform_pos=rel_pos,
                        input_kd=color[:3],
                        scale=np.array(dimensions),
                        overwrite_material=overwrite_material)
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))
                                        ] = len(self.renderer.visual_objects) - 1
                visual_objects.append(
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(
                    igibson.assets_path, 'models/mjcf_primitives/sphere8.obj')
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
                    igibson.assets_path, 'models/mjcf_primitives/cube.obj')
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
                    igibson.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(len(self.renderer.visual_objects) - 1)
                link_ids.append(link_id)

            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(object_pb_id)
            else:
                _, _, _, _, pos, orn = p.getLinkState(object_pb_id, link_id)
            poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        self.renderer.add_instance_group(object_ids=visual_objects,
                                         link_ids=link_ids,
                                         pybullet_uuid=object_pb_id,
                                         class_id=class_id,
                                         poses_trans=poses_trans,
                                         poses_rot=poses_rot,
                                         dynamic=True,
                                         robot=None,
                                         use_pbr=use_pbr,
                                         use_pbr_mapping=use_pbr_mapping,
                                         shadow_caster=shadow_caster)

    @load_without_pybullet_vis
    def import_robot(self,
                     robot,
                     class_id=SemanticClass.ROBOTS):
        """
        Import a robot into the simulator

        :param robot: Robot
        :param class_id: Class id for rendering semantic segmentation
        :return: pybullet id
        """
        assert isinstance(robot, BaseRobot), \
            'import_robot can only be called with BaseRobot'
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
                if (filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn)) not in self.visual_objects.keys():
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))
                                        ] = len(self.renderer.visual_objects) - 1
                visual_objects.append(
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))])
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(
                    igibson.assets_path, 'models/mjcf_primitives/sphere8.obj')
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
                    igibson.assets_path, 'models/mjcf_primitives/cube.obj')
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
                    igibson.assets_path, 'models/mjcf_primitives/cube.obj')
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

        self.renderer.add_robot(object_ids=visual_objects,
                                link_ids=link_ids,
                                pybullet_uuid=ids[0],
                                class_id=class_id,
                                poses_rot=poses_rot,
                                poses_trans=poses_trans,
                                dynamic=True,
                                robot=robot)

        return ids

    def _step_simulation(self):
        """
        Step the simulation for one step and update positions in renderer
        """
        p.stepSimulation()
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.update_position(instance)

    def step(self):
        """
        Step the simulation at self.render_timestep and update positions in renderer
        """
        for _ in range(int(self.render_timestep / self.physics_timestep)):
            p.stepSimulation()
        self.sync()

    def sync(self):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        self.body_links_awake = 0
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.body_links_awake += self.update_position(instance)
        if self.use_ig_renderer and self.viewer is not None:
            self.viewer.update()
        if self.first_sync:
            self.first_sync = False

    def update_position(self, instance):
        """
        Update position for an object or a robot in renderer.

        :param instance: Instance in the renderer
        """
        body_links_awake = 0
        if isinstance(instance, Instance):
            dynamics_info = p.getDynamicsInfo(instance.pybullet_uuid, -1)
            inertial_pos = dynamics_info[3]
            inertial_orn = dynamics_info[4]
            if len(dynamics_info) == 13 and not self.first_sync:
                activation_state = dynamics_info[12]
            else:
                activation_state = PyBulletSleepState.AWAKE

            if activation_state != PyBulletSleepState.AWAKE:
                return body_links_awake
            # pos and orn of the inertial frame of the base link,
            # instead of the base link frame
            pos, orn = p.getBasePositionAndOrientation(
                instance.pybullet_uuid)

            # Need to convert to the base link frame because that is
            # what our own renderer keeps track of
            # Based on pyullet docuementation:
            # urdfLinkFrame = comLinkFrame * localInertialFrame.inverse().

            inv_inertial_pos, inv_inertial_orn =\
                p.invertTransform(inertial_pos, inertial_orn)
            # Now pos and orn are converted to the base link frame
            pos, orn = p.multiplyTransforms(
                pos, orn, inv_inertial_pos, inv_inertial_orn)

            instance.set_position(pos)
            instance.set_rotation(quat2rotmat(xyzw2wxyz(orn)))
            body_links_awake += 1
        elif isinstance(instance, InstanceGroup):
            for j, link_id in enumerate(instance.link_ids):
                if link_id == -1:
                    dynamics_info = p.getDynamicsInfo(
                        instance.pybullet_uuid, -1)
                    inertial_pos = dynamics_info[3]
                    inertial_orn = dynamics_info[4]
                    if len(dynamics_info) == 13 and not self.first_sync:
                        activation_state = dynamics_info[12]
                    else:
                        activation_state = PyBulletSleepState.AWAKE

                    if activation_state != PyBulletSleepState.AWAKE:
                        continue
                    # same conversion is needed as above
                    pos, orn = p.getBasePositionAndOrientation(
                        instance.pybullet_uuid)

                    inv_inertial_pos, inv_inertial_orn =\
                        p.invertTransform(inertial_pos, inertial_orn)
                    pos, orn = p.multiplyTransforms(
                        pos, orn, inv_inertial_pos, inv_inertial_orn)
                else:
                    dynamics_info = p.getDynamicsInfo(
                        instance.pybullet_uuid, link_id)

                    if len(dynamics_info) == 13 and not self.first_sync:
                        activation_state = dynamics_info[12]
                    else:
                        activation_state = PyBulletSleepState.AWAKE

                    if activation_state != PyBulletSleepState.AWAKE:
                        continue
                    _, _, _, _, pos, orn = p.getLinkState(
                        instance.pybullet_uuid, link_id)

                instance.set_position_for_part(xyz2mat(pos), j)
                instance.set_rotation_for_part(
                    quat2rotmat(xyzw2wxyz(orn)), j)
                body_links_awake += 1
        return body_links_awake

    def isconnected(self):
        """
        :return: pybullet is alive
        """
        return p.getConnectionInfo(self.cid)['isConnected']

    def disconnect(self):
        """
        Clean up the simulator
        """
        if self.isconnected():
            print("******************PyBullet Logging Information:")
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)
            print("PyBullet Logging Information******************")
        self.renderer.release()
