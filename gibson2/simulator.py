from gibson2.utils.mesh_util import quat2rotmat, xyzw2wxyz, xyz2mat
from gibson2.utils.semantics_utils import get_class_name_to_class_id
from gibson2.utils.constants import SemanticClass, PyBulletSleepState
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR, VrSettings
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from gibson2.render.mesh_renderer.instances import InstanceGroup, Instance, Robot
from gibson2.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.render.viewer import Viewer, ViewerVR, ViewerSimple
from gibson2.objects.articulated_object import ArticulatedObject, URDFObject
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.scenes.scene_base import Scene
from gibson2.robots.robot_base import BaseRobot
from gibson2.objects.object_base import Object


import pybullet as p
import gibson2
import os
import numpy as np
import platform
import logging
import time


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
                 rendering_settings=MeshRendererSettings(),
                 vr_settings=VrSettings()):
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
        disable it when you want to run multiple physics step but don't need to visualize each frame
        :param rendering_settings: settings to use for mesh renderer
        :param vr_settings: settings to use for VR in simulator and MeshRendererVR
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
        self.use_vr_renderer = False
        self.use_simple_viewer = False

        if self.mode in ['gui', 'iggui']:
            self.use_ig_renderer = True

        if self.mode in ['gui', 'pbgui']:
            self.use_pb_renderer = True

        if self.mode in ['vr']:
            self.use_vr_renderer = True

        if self.mode in ['simple']:
            self.use_simple_viewer = True

        # Starting position for the VR (default set to None if no starting position is specified by the user)
        self.vr_start_pos = None
        self.max_haptic_duration = 4000
        self.image_width = image_width
        self.image_height = image_height
        self.vertical_fov = vertical_fov
        self.device_idx = device_idx
        self.render_to_tensor = render_to_tensor

        self.optimized_renderer = rendering_settings.optimized
        self.rendering_settings = rendering_settings
        self.viewer = None
        self.vr_settings = vr_settings
        # We must be using the Simulator's vr mode and have use_vr set to true in the settings to access the VR context
        self.can_access_vr_context = self.use_vr_renderer and self.vr_settings.use_vr

        # Settings for adjusting physics and render timestep in vr
        # Fraction to multiple previous render timestep by in low-pass filter
        self.lp_filter_frac = 0.9

        # Variables for data saving and replay in VR
        self.last_physics_timestep = -1
        self.last_render_timestep = -1
        self.last_physics_step_num = -1
        self.last_frame_dur = -1

        self.load()

        self.class_name_to_class_id = get_class_name_to_class_id()
        self.body_links_awake = 0

    def set_timestep(self, physics_timestep, render_timestep):
        """
        Set physics timestep and render (action) timestep

        :param physics_timestep: physics timestep for pybullet
        :param render_timestep: rendering timestep for renderer
        """
        self.physics_timestep = physics_timestep
        self.render_timestep = render_timestep
        p.setTimeStep(self.physics_timestep)

    def set_render_timestep(self, render_timestep):
        """
        :param render_timestep: render timestep to set in the Simulator
        """
        self.render_timestep = render_timestep

    def add_viewer(self):
        """
        Attach a debugging viewer to the renderer.
        This will make the step much slower so should be avoided when training agents
        """
        if self.use_vr_renderer:
            self.viewer = ViewerVR()
        elif self.use_simple_viewer:
            self.viewer = ViewerSimple()
        else:
            self.viewer = Viewer(simulator=self, renderer=self.renderer)
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
                                            rendering_settings=self.rendering_settings)
        elif self.use_vr_renderer:
            self.renderer = MeshRendererVR(
                rendering_settings=self.rendering_settings, vr_settings=self.vr_settings)
        else:
            self.renderer = MeshRenderer(width=self.image_width,
                                         height=self.image_height,
                                         vertical_fov=self.vertical_fov,
                                         device_idx=self.device_idx,
                                         rendering_settings=self.rendering_settings)

        # print("******************PyBullet Logging Information:")
        if self.use_pb_renderer:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)
        p.setTimeStep(self.physics_timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        # print("PyBullet Logging Information******************")

        self.visual_objects = {}
        self.robots = []
        self.scene = None
        if (self.use_ig_renderer or self.use_vr_renderer or self.use_simple_viewer) and not self.render_to_tensor:
            self.add_viewer()

    def optimize_vertex_and_texture(self):
        self.renderer.optimize_vertex_and_texture()

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
                    class_id=body_id,
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
        if isinstance(obj, ArticulatedObject) or isinstance(obj, URDFObject):
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
                if (filename, (*dimensions)) not in self.visual_objects.keys():
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions),
                                              texture_scale=texture_scale,
                                              load_texture=load_texture)
                    self.visual_objects[(filename, (*dimensions))
                                        ] = len(self.renderer.visual_objects) - 1
                visual_object = self.visual_objects[(filename, (*dimensions))]
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/sphere8.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5])
                visual_object = len(self.renderer.get_visual_objects()) - 1
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]])
                visual_object = len(self.renderer.get_visual_objects()) - 1
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
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
                        gibson2.assets_path,
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
                if (filename, (*dimensions)) not in self.visual_objects.keys():
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
                    self.visual_objects[(filename, (*dimensions))
                                        ] = len(self.renderer.visual_objects) - 1
                visual_objects.append(
                    self.visual_objects[(filename, (*dimensions))])
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
                visual_objects.append(
                    len(self.renderer.get_visual_objects()) - 1)
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
                visual_objects.append(
                    len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(
                    len(self.renderer.get_visual_objects()) - 1)
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
                if (filename, (*dimensions)) not in self.visual_objects.keys():
                    self.renderer.load_object(filename,
                                              transform_orn=rel_orn,
                                              transform_pos=rel_pos,
                                              input_kd=color[:3],
                                              scale=np.array(dimensions))
                    self.visual_objects[(filename, (*dimensions))
                                        ] = len(self.renderer.visual_objects) - 1
                visual_objects.append(
                    self.visual_objects[(filename, (*dimensions))])
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
                visual_objects.append(
                    len(self.renderer.get_visual_objects()) - 1)
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
                visual_objects.append(
                    len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(
                    len(self.renderer.get_visual_objects()) - 1)
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

    def step(self, print_time=False, use_render_timestep_lpf=True):
        """
        Step the simulation at self.render_timestep and update positions in renderer
        """
        # First poll VR events and store them
        if self.can_access_vr_context:
            # Note: this should only be called once per frame - use get_vr_events to read the event data list in
            # subsequent read operations
            self.poll_vr_events()

        physics_start_time = time.time()
        physics_timestep_num = int(
            self.render_timestep / self.physics_timestep)
        for _ in range(physics_timestep_num):
            p.stepSimulation()
        self.sync()
        physics_dur = time.time() - physics_start_time

        render_start_time = time.time()
        render_dur = time.time() - render_start_time
        # Update render timestep using low-pass filter function
        if use_render_timestep_lpf:
            self.render_timestep = self.lp_filter_frac * \
                self.render_timestep + (1 - self.lp_filter_frac) * render_dur
        frame_dur = physics_dur + render_dur

        # Set variables for data saving and replay
        self.last_physics_timestep = physics_dur
        self.last_render_timestep = render_dur
        self.last_physics_step_num = physics_timestep_num
        self.last_frame_dur = frame_dur

        # Sets the VR starting position if one has been specified by the user
        self.perform_vr_start_pos_move()

        if print_time:
            print('Total frame duration: {} and FPS: {}'.format(
                round(frame_dur, 2), round(1/max(frame_dur, 0.002), 2)))
            print('Total physics duration: {} and FPS: {}'.format(
                round(physics_dur, 2), round(1/max(physics_dur, 0.002), 2)))
            print('Number of 1/120 physics steps: {}'.format(physics_timestep_num))
            print('Total render duration: {} and FPS: {}'.format(
                round(render_dur, 2), round(1/max(render_dur, 0.002), 2)))
            print('-------------------------')

    def sync(self):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        self.body_links_awake = 0
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.body_links_awake += self.update_position(instance)
        if (self.use_ig_renderer or self.use_vr_renderer or self.use_simple_viewer) and self.viewer is not None:
            self.viewer.update()

    # Sets the VR position on the first step iteration where the hmd tracking is valid. Not to be confused
    # with self.set_vr_start_pos, which simply records the desired start position before the simulator starts running.
    def perform_vr_start_pos_move(self):
        # Update VR start position if it is not None and the hmd is valid
        # This will keep checking until we can successfully set the start position
        if self.vr_start_pos:
            hmd_is_valid, _, _, _ = self.renderer.vrsys.getDataForVRDevice(
                'hmd')
            if hmd_is_valid:
                offset_to_start = np.array(
                    self.vr_start_pos) - self.get_hmd_world_pos()
                if self.vr_height_offset is not None:
                    offset_to_start[2] = self.vr_height_offset
                self.set_vr_offset(offset_to_start)
                self.vr_start_pos = None

    # Returns VR event data as list of lists. Each sub-list contains deviceType and eventType.
    # List is empty if all events are invalid.
    # deviceType: left_controller, right_controller
    # eventType: grip_press, grip_unpress, trigger_press, trigger_unpress, touchpad_press, touchpad_unpress,
    # touchpad_touch, touchpad_untouch, menu_press, menu_unpress (menu is the application button)
    def poll_vr_events(self):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        self.vr_event_data = self.renderer.vrsys.pollVREvents()
        return self.vr_event_data

    # Returns the VR events processed by the simulator
    def get_vr_events(self):
        return self.vr_event_data

    # Queries system for a VR event, and returns true if that event happened this frame
    def query_vr_event(self, device, event):
        for ev_data in self.vr_event_data:
            if device == ev_data[0] and event == ev_data[1]:
                return True

        return False

    # Call this after step - returns all VR device data for a specific device
    # Device can be hmd, left_controller or right_controller
    # Returns isValid (indicating validity of data), translation and rotation in Gibson world space
    def get_data_for_vr_device(self, deviceName):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        # Use fourth variable in list to get actual hmd position in space
        is_valid, translation, rotation, _ = self.renderer.vrsys.getDataForVRDevice(
            deviceName)
        return [is_valid, translation, rotation]

    # Get world position of HMD without offset
    def get_hmd_world_pos(self):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        _, _, _, hmd_world_pos = self.renderer.vrsys.getDataForVRDevice('hmd')
        return hmd_world_pos

    # Call this after getDataForVRDevice - returns analog data for a specific controller
    # Controller can be left_controller or right_controller
    # Returns trigger_fraction, touchpad finger position x, touchpad finger position y
    # Data is only valid if isValid is true from previous call to getDataForVRDevice
    # Trigger data: 1 (closed) <------> 0 (open)
    # Analog data: X: -1 (left) <-----> 1 (right) and Y: -1 (bottom) <------> 1 (top)
    def get_button_data_for_controller(self, controllerName):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        trigger_fraction, touch_x, touch_y = self.renderer.vrsys.getButtonDataForController(
            controllerName)
        return [trigger_fraction, touch_x, touch_y]

    # Returns eye tracking data as list of lists. Order: is_valid, gaze origin, gaze direction, gaze point, left pupil diameter, right pupil diameter (both in millimeters)
    # Call after getDataForVRDevice, to guarantee that latest HMD transform has been acquired
    def get_eye_tracking_data(self):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = self.renderer.vrsys.getEyeTrackingData()
        return [is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter]

    # Sets the starting position of the VR system in iGibson space
    def set_vr_start_pos(self, start_pos=None, vr_height_offset=None):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        # The VR headset will actually be set to this position during the first frame.
        # This is because we need to know where the headset is in space when it is first picked
        # up to set the initial offset correctly.
        self.vr_start_pos = start_pos
        # This value can be set to specify a height offset instead of an absolute height.
        # We might want to adjust the height of the camera based on the height of the person using VR,
        # but still offset this height. When this option is not None it offsets the height by the amount
        # specified instead of overwriting the VR system height output.
        self.vr_height_offset = vr_height_offset

    # Sets the world position of the VR system in iGibson space
    def set_vr_pos(self, pos=None):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        offset_to_pos = np.array(pos) - self.get_hmd_world_pos()
        self.set_vr_offset(offset_to_pos)

    # Gets the world position of the VR system in iGibson space
    def get_vr_pos(self):
        return self.get_hmd_world_pos() + self.get_vr_offset()

    # Sets the translational offset of the VR system (HMD, left controller, right controller) from world space coordinates
    # Can be used for many things, including adjusting height and teleportation-based movement
    # Input must be a list of three floats, corresponding to x, y, z in Gibson coordinate space
    def set_vr_offset(self, pos=None):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        self.renderer.vrsys.setVROffset(-pos[1], pos[2], -pos[0])

    # Gets the current VR offset vector in list form: x, y, z (in Gibson coordinates)
    def get_vr_offset(self):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        x, y, z = self.renderer.vrsys.getVROffset()
        return [x, y, z]

    # Gets the direction vectors representing the device's coordinate system in list form: x, y, z (in Gibson coordinates)
    # List contains "right", "up" and "forward" vectors in that order
    # Device can be one of "hmd", "left_controller" or "right_controller"
    def get_device_coordinate_system(self, device):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        vec_list = []

        coordinate_sys = self.renderer.vrsys.getDeviceCoordinateSystem(device)
        for dir_vec in coordinate_sys:
            vec_list.append(dir_vec)

        return vec_list

    # Triggers a haptic pulse of the specified strength (0 is weakest, 1 is strongest)
    # Device can be one of "hmd", "left_controller" or "right_controller"
    def trigger_haptic_pulse(self, device, strength):
        if not self.can_access_vr_context:
            raise RuntimeError(
                'ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!')

        self.renderer.vrsys.triggerHapticPulseForDevice(
            device, int(self.max_haptic_duration * strength))

    # Note: this function must be called after optimize_vertex_and_texture is called
    # Note: this function currently only works with the optimized renderer - please use the renderer hidden list
    # to hide objects in the non-optimized renderer
    def set_hidden_state(self, obj, hide=True):
        """
        Sets the hidden state of an object to be either hidden or not hidden.
        The object passed in must inherent from Object at the top level.
        """
        # Find instance corresponding to this id in the renderer
        for instance in self.renderer.instances:
            if obj.body_id == instance.pybullet_uuid:
                instance.hidden = hide
                self.renderer.update_hidden_state([instance])
                return

    def get_hidden_state(self, obj):
        """
        Returns the current hidden state of the object - hidden (True) or not hidden (False).
        """
        for instance in self.renderer.instances:
            if obj.body_id == instance.pybullet_uuid:
                return instance.hidden

    def get_floor_ids(self):
        """
        Gets the body ids for all floor objects in the scene. This is used internally
        by the VrBody class to disable collisions with the floor.
        """
        if not hasattr(self.scene, 'objects_by_id'):
            return []
        floor_ids = []
        for body_id in self.objects:
            if body_id in self.scene.objects_by_id.keys() and self.scene.objects_by_id[body_id].category == 'floors':
                floor_ids.append(body_id)
        return floor_ids

    @staticmethod
    def update_position(instance):
        """
        Update position for an object or a robot in renderer.
        :param instance: Instance in the renderer
        """
        body_links_awake = 0
        if isinstance(instance, Instance):
            dynamics_info = p.getDynamicsInfo(instance.pybullet_uuid, -1)
            inertial_pos = dynamics_info[3]
            inertial_orn = dynamics_info[4]
            if len(dynamics_info) == 13:
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
                    if len(dynamics_info) == 13:
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

                    if len(dynamics_info) == 13:
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
            # print("******************PyBullet Logging Information:")
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)
            # print("PyBullet Logging Information******************")
        self.renderer.release()

    def disconnect_pybullet(self):
        """
        Disconnects only pybullet - used for multi-user VR
        """
        if self.isconnected():
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)
