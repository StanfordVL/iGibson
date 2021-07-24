import ctypes
import logging
import os
import platform
import time
from time import sleep

import numpy as np
import pybullet as p

import igibson
from igibson.object_states.factory import get_states_by_dependency_order
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.objects.object_base import Object
from igibson.objects.particles import Particle, ParticleSystem
from igibson.objects.stateful_object import StatefulObject
from igibson.objects.visual_marker import VisualMarker
from igibson.objects.visual_shape import VisualShape
from igibson.render.mesh_renderer.instances import Instance, InstanceGroup
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR, VrSettings
from igibson.render.viewer import Viewer, ViewerSimple, ViewerVR
from igibson.robots.robot_base import BaseRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.scene_base import Scene
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.utils.constants import PyBulletSleepState, SemanticClass
from igibson.utils.mesh_util import quat2rotmat, xyz2mat, xyzw2wxyz
from igibson.utils.semantics_utils import get_class_name_to_class_id
from igibson.utils.utils import quatXYZWFromRotMat
from igibson.utils.vr_utils import VR_CONTROLLERS, VR_DEVICES, VrData, calc_offset, calc_z_rot_from_right


class Simulator:
    """
    Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
    both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.
    """

    def __init__(
        self,
        gravity=9.8,
        physics_timestep=1 / 120.0,
        render_timestep=1 / 30.0,
        solver_iterations=100,
        mode="gui",
        image_width=128,
        image_height=128,
        vertical_fov=90,
        device_idx=0,
        render_to_tensor=False,
        rendering_settings=MeshRendererSettings(),
        vr_settings=VrSettings(),
    ):
        """
        :param gravity: gravity on z direction.
        :param physics_timestep: timestep of physical simulation, p.stepSimulation()
        :param render_timestep: timestep of rendering, and Simulator.step() function
        :param solver_iterations: number of solver iterations to feed into pybullet, can be reduced to increase speed.
            pybullet default value is 50.
        :param use_variable_step_num: whether to use a fixed (1) or variable physics step number
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
        self.solver_iterations = solver_iterations
        self.physics_timestep_num = self.render_timestep / self.physics_timestep
        assert self.physics_timestep_num.is_integer(), "render_timestep must be a multiple of physics_timestep"
        self.physics_timestep_num = int(self.physics_timestep_num)

        self.mode = mode

        self.scene = None

        self.particle_systems = []

        # TODO: remove this, currently used for testing only
        self.objects = []

        plt = platform.system()
        if plt == "Darwin" and self.mode == "gui":
            self.mode = "iggui"  # for mac os disable pybullet rendering
            logging.warn(
                "Rendering both iggui and pbgui is not supported on mac, choose either pbgui or "
                "iggui. Default to iggui."
            )
        if plt == "Windows" and self.mode in ["vr"]:
            # By default, windows does not provide ms level timing accuracy
            winmm = ctypes.WinDLL("winmm")
            winmm.timeBeginPeriod(1)

        self.use_pb_renderer = False
        self.use_ig_renderer = False
        self.use_vr_renderer = False
        self.use_simple_viewer = False

        if self.mode in ["gui", "iggui"]:
            self.use_ig_renderer = True

        if self.mode in ["gui", "pbgui"]:
            self.use_pb_renderer = True

        if self.mode in ["vr"]:
            self.use_vr_renderer = True
            rendering_settings.blend_highlight = True

        if self.mode in ["simple"]:
            self.use_simple_viewer = True

        # Starting position for the VR (default set to None if no starting position is specified by the user)
        self.vr_start_pos = None
        self.eye_tracking_data = None
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
        self.vr_overlay_initialized = False
        # We must be using the Simulator's vr mode and have use_vr set to true in the settings to access the VR context
        self.can_access_vr_context = self.use_vr_renderer and self.vr_settings.use_vr
        # Duration of a vsync frame - assumes 90Hz refresh rate
        self.vsync_frame_dur = 11.11e-3
        # Get expected number of vsync frames per iGibson frame
        # Note: currently assumes a 90Hz VR system
        self.vsync_frame_num = int(round(self.render_timestep / self.vsync_frame_dur))
        # Total amount of time we want non-blocking actions to take each frame
        # Leave a small amount of time before the last vsync, just in case we overrun
        self.non_block_frame_time = (self.vsync_frame_num - 1) * self.vsync_frame_dur + (
            5e-3 if self.vr_settings.curr_device == "OCULUS" else 10e-3
        )
        # Timing variables for functions called outside of step() that also take up frame time
        self.frame_end_time = None
        self.main_vr_robot = None

        # Variables for data saving and replay in VR
        self.last_physics_timestep = -1
        self.last_render_timestep = -1
        self.last_physics_step_num = -1
        self.last_frame_dur = -1
        self.frame_count = 0

        self.load()

        self.class_name_to_class_id = get_class_name_to_class_id()
        self.body_links_awake = 0
        # First sync always sync all objects (regardless of their sleeping states)
        self.first_sync = True
        # Set of categories that can be grasped by assisted grasping
        self.assist_grasp_category_allow_list = set()
        self.gen_assisted_grasping_categories()
        self.assist_grasp_mass_thresh = 10.0

        self.object_state_types = get_states_by_dependency_order()

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
            self.viewer = ViewerVR(
                self.vr_settings.use_companion_window, frame_save_path=self.vr_settings.frame_save_path
            )
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
            self.renderer = MeshRendererG2G(
                width=self.image_width,
                height=self.image_height,
                vertical_fov=self.vertical_fov,
                device_idx=self.device_idx,
                rendering_settings=self.rendering_settings,
                simulator=self,
            )
        elif self.use_vr_renderer:
            self.renderer = MeshRendererVR(
                rendering_settings=self.rendering_settings, vr_settings=self.vr_settings, simulator=self
            )
        else:
            self.renderer = MeshRenderer(
                width=self.image_width,
                height=self.image_height,
                vertical_fov=self.vertical_fov,
                device_idx=self.device_idx,
                rendering_settings=self.rendering_settings,
                simulator=self,
            )

        # print("******************PyBullet Logging Information:")
        if self.use_pb_renderer:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        # Simulation reset is needed for deterministic action replay
        if self.vr_settings.reset_sim:
            p.resetSimulation()
            p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        p.setPhysicsEngineParameter(numSolverIterations=self.solver_iterations)
        p.setTimeStep(self.physics_timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        self.visual_objects = {}
        self.robots = []
        self.scene = None
        if (self.use_ig_renderer or self.use_vr_renderer or self.use_simple_viewer) and not self.render_to_tensor:
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
    def import_scene(
        self,
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
        assert isinstance(scene, Scene) and not isinstance(
            scene, InteractiveIndoorScene
        ), "import_scene can only be called with Scene that is not InteractiveIndoorScene"
        # Load the scene. Returns a list of pybullet ids of the objects loaded that we can use to
        # load them in the renderer
        new_object_pb_ids = scene.load()
        self.objects += new_object_pb_ids

        # Load the objects in the renderer
        for new_object_pb_id in new_object_pb_ids:
            self.load_object_in_renderer(
                new_object_pb_id,
                class_id=class_id,
                texture_scale=texture_scale,
                load_texture=load_texture,
                render_floor_plane=render_floor_plane,
                use_pbr=False,
                use_pbr_mapping=False,
            )

            # TODO: add instance renferencing for iG v1 scenes

        self.scene = scene

        # Load the states of all the objects in the scene.
        for obj in scene.get_objects():
            if isinstance(obj, StatefulObject):
                if isinstance(obj, ObjectMultiplexer):
                    for sub_obj in obj._multiplexed_objects:
                        if isinstance(sub_obj, ObjectGrouper):
                            for group_sub_obj in sub_obj.objects:
                                for state in group_sub_obj.states.values():
                                    state.initialize(self)
                        else:
                            for state in sub_obj.states.values():
                                state.initialize(self)
                else:
                    for state in obj.states.values():
                        state.initialize(self)

        return new_object_pb_ids

    @load_without_pybullet_vis
    def import_ig_scene(self, scene):
        """
        Import scene from iGSDF class

        :param scene: iGSDFScene instance
        :return: pybullet body ids from scene.load function
        """
        assert isinstance(
            scene, InteractiveIndoorScene
        ), "import_ig_scene can only be called with InteractiveIndoorScene"
        if not self.use_pb_renderer:
            scene.set_ignore_visual_shape(True)
            # skip loading visual shape if not using pybullet visualizer

        new_object_ids = scene.load()
        self.objects += new_object_ids

        for body_id, visual_mesh_to_material, link_name_to_vm in zip(
            new_object_ids, scene.visual_mesh_to_material, scene.link_name_to_vm
        ):
            use_pbr = True
            use_pbr_mapping = True
            shadow_caster = True
            physical_object = scene.objects_by_id[body_id]
            if scene.scene_source == "IG":
                if physical_object.category in ["walls", "floors", "ceilings"]:
                    use_pbr = False
                    use_pbr_mapping = False
                if physical_object.category == "ceilings":
                    shadow_caster = False
            class_id = self.class_name_to_class_id.get(physical_object.category, SemanticClass.SCENE_OBJS)
            self.load_articulated_object_in_renderer(
                body_id,
                class_id=class_id,
                visual_mesh_to_material=visual_mesh_to_material,
                link_name_to_vm=link_name_to_vm,
                use_pbr=use_pbr,
                use_pbr_mapping=use_pbr_mapping,
                shadow_caster=shadow_caster,
                physical_object=physical_object,
            )

        self.scene = scene

        # Load the states of all the objects in the scene.
        for obj in scene.get_objects():
            if isinstance(obj, StatefulObject):
                if isinstance(obj, ObjectMultiplexer):
                    for sub_obj in obj._multiplexed_objects:
                        if isinstance(sub_obj, ObjectGrouper):
                            for group_sub_obj in sub_obj.objects:
                                for state in group_sub_obj.states.values():
                                    state.initialize(self)
                        else:
                            for state in sub_obj.states.values():
                                state.initialize(self)
                else:
                    for state in obj.states.values():
                        state.initialize(self)

        return new_object_ids

    @load_without_pybullet_vis
    def import_particle_system(self, obj):
        """
        Import an object into the simulator
        :param obj: ParticleSystem to load
        """

        assert isinstance(obj, ParticleSystem), "import_particle_system can only be called with ParticleSystem"

        self.particle_systems.append(obj)
        obj.initialize(self)

    @load_without_pybullet_vis
    def import_object(
        self, obj, class_id=SemanticClass.USER_ADDED_OBJS, use_pbr=True, use_pbr_mapping=True, shadow_caster=True
    ):
        """
        Import an object into the simulator

        :param obj: Object to load
        :param class_id: Class id for rendering semantic segmentation
        :param use_pbr: Whether to use pbr
        :param use_pbr_mapping: Whether to use pbr mapping
        :param shadow_caster: Whether to cast shadow
        """
        assert isinstance(obj, Object), "import_object can only be called with Object"

        if isinstance(obj, VisualMarker) or isinstance(obj, VisualShape) or isinstance(obj, Particle):
            # Marker objects can be imported without a scene.
            new_object_pb_id_or_ids = obj.load()
        else:
            # Non-marker objects require a Scene to be imported.
            assert self.scene is not None, "A scene must be imported before additional objects can be imported."
            # Load the object in pybullet. Returns a pybullet id that we can use to load it in the renderer
            new_object_pb_id_or_ids = self.scene.add_object(obj, _is_call_from_simulator=True)

        # If no new bodies are immediately imported into pybullet, we have no rendering steps.
        if new_object_pb_id_or_ids is None:
            return None

        if isinstance(new_object_pb_id_or_ids, list):
            new_object_pb_ids = new_object_pb_id_or_ids
        else:
            new_object_pb_ids = [new_object_pb_id_or_ids]
        self.objects += new_object_pb_ids

        for i, new_object_pb_id in enumerate(new_object_pb_ids):
            if isinstance(obj, ArticulatedObject) or isinstance(obj, URDFObject):
                if isinstance(obj, ArticulatedObject):
                    visual_mesh_to_material = None
                else:
                    visual_mesh_to_material = obj.visual_mesh_to_material[i]
                link_name_to_vm = obj.link_name_to_vm[i]
                self.load_articulated_object_in_renderer(
                    new_object_pb_id,
                    class_id=class_id,
                    use_pbr=use_pbr,
                    use_pbr_mapping=use_pbr_mapping,
                    visual_mesh_to_material=visual_mesh_to_material,
                    link_name_to_vm=link_name_to_vm,
                    shadow_caster=shadow_caster,
                    physical_object=obj,
                )
            else:
                softbody = obj.__class__.__name__ == "SoftObject"
                self.load_object_in_renderer(
                    new_object_pb_id,
                    class_id=class_id,
                    softbody=softbody,
                    use_pbr=use_pbr,
                    use_pbr_mapping=use_pbr_mapping,
                    shadow_caster=shadow_caster,
                    physical_object=obj,
                )

        # Finally, initialize the object's states
        if isinstance(obj, StatefulObject):
            if isinstance(obj, ObjectMultiplexer):
                for sub_obj in obj._multiplexed_objects:
                    if isinstance(sub_obj, ObjectGrouper):
                        for group_sub_obj in sub_obj.objects:
                            for state in group_sub_obj.states.values():
                                state.initialize(self)
                    else:
                        for state in sub_obj.states.values():
                            state.initialize(self)
            else:
                for state in obj.states.values():
                    state.initialize(self)

        return new_object_pb_id_or_ids

    @load_without_pybullet_vis
    def load_visual_sphere(self, radius, color=[1, 0, 0]):
        """
        Load a visual-only (not controlled by pybullet) sphere into the renderer.
        Such a sphere can be moved around without affecting PyBullet determinism.
        :param radius: the radius of the visual sphere in meters
        :param color: RGB color of sphere (from 0 to 1 on each axis)
        """
        sphere_file = os.path.join(igibson.assets_path, "models/mjcf_primitives/sphere8.obj")
        self.renderer.load_object(
            sphere_file,
            transform_orn=[0, 0, 0, 1],
            transform_pos=[0, 0, 0],
            input_kd=[1, 0, 0],
            scale=[radius, radius, radius],
        )
        visual_object = len(self.renderer.get_visual_objects()) - 1
        self.renderer.add_instance(
            visual_object,
            pybullet_uuid=0,  # this can be ignored
            class_id=1,  # this can be ignored
            dynamic=False,
            softbody=False,
            use_pbr=False,
            use_pbr_mapping=False,
            shadow_caster=False,
        )
        # Return instance so we can control it
        return self.renderer.instances[-1]

    @load_without_pybullet_vis
    def load_object_in_renderer(
        self,
        object_pb_id,
        class_id=None,
        softbody=False,
        texture_scale=1.0,
        load_texture=True,
        render_floor_plane=False,
        use_pbr=True,
        use_pbr_mapping=True,
        shadow_caster=True,
        physical_object=None,
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
        :param physical_object: The reference to Object class
        """

        # Load object in renderer, use visual shape and base_link frame
        # not CoM frame
        # Do not load URDFObject or ArticulatedObject with this function
        if physical_object is not None and (
            isinstance(physical_object, ArticulatedObject) or isinstance(physical_object, URDFObject)
        ):
            raise ValueError("loading articulated object with load_object_in_renderer function")

        for shape in p.getVisualShapeData(object_pb_id):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            dynamics_info = p.getDynamicsInfo(id, link_id)
            inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
            rel_pos, rel_orn = p.multiplyTransforms(*p.invertTransform(inertial_pos, inertial_orn), rel_pos, rel_orn)
            # visual meshes frame are transformed from the urdfLinkFrame as origin to comLinkFrame as origin
            visual_object = None
            if type == p.GEOM_MESH:
                filename = filename.decode("utf-8")
                if (filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn)) not in self.visual_objects.keys():
                    self.renderer.load_object(
                        filename,
                        transform_orn=rel_orn,
                        transform_pos=rel_pos,
                        input_kd=color[:3],
                        scale=np.array(dimensions),
                        texture_scale=texture_scale,
                        load_texture=load_texture,
                    )
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))] = (
                        len(self.renderer.visual_objects) - 1
                    )
                visual_object = self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))]
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/sphere8.obj")
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5],
                )
                visual_object = len(self.renderer.get_visual_objects()) - 1
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/cube.obj")
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]],
                )
                visual_object = len(self.renderer.get_visual_objects()) - 1
            elif type == p.GEOM_BOX:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/cube.obj")
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=np.array(dimensions),
                )
                visual_object = len(self.renderer.visual_objects) - 1
            elif type == p.GEOM_PLANE:
                # By default, we add an additional floor surface to "smooth out" that of the original mesh.
                # Normally you don't need to render this additionally added floor surface.
                # However, if you do want to render it for some reason, you can set render_floor_plane to be True.
                if render_floor_plane:
                    filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/cube.obj")
                    self.renderer.load_object(
                        filename,
                        transform_orn=rel_orn,
                        transform_pos=rel_pos,
                        input_kd=color[:3],
                        scale=[100, 100, 0.01],
                    )
                    visual_object = len(self.renderer.visual_objects) - 1
            if visual_object is not None:
                self.renderer.add_instance(
                    visual_object,
                    pybullet_uuid=object_pb_id,
                    class_id=class_id,
                    dynamic=True,
                    softbody=softbody,
                    use_pbr=use_pbr,
                    use_pbr_mapping=use_pbr_mapping,
                    shadow_caster=shadow_caster,
                )
                if physical_object is not None:
                    physical_object.renderer_instances.append(self.renderer.instances[-1])

    @load_without_pybullet_vis
    def load_articulated_object_in_renderer(
        self,
        object_pb_id,
        physical_object,
        link_name_to_vm,
        class_id=None,
        visual_mesh_to_material=None,
        use_pbr=True,
        use_pbr_mapping=True,
        shadow_caster=True,
    ):
        """
        Load the articulated object into renderer

        :param object_pb_id: pybullet body id
        :param physical_object: The reference to Object class
        :param link_name_to_vm: mapping from link name to a list of visual mesh file paths
        :param class_id: Class id for rendering semantic segmentation
        :param visual_mesh_to_material: mapping from visual mesh to randomizable materials
        :param use_pbr: Whether to use pbr
        :param use_pbr_mapping: Whether to use pbr mapping
        :param shadow_caster: Whether to cast shadow
        """
        # Load object in renderer, use visual shape from physical_object class
        # using CoM frame
        # only load URDFObject or ArticulatedObject with this function
        if not (
            isinstance(physical_object, ArticulatedObject)
            or isinstance(physical_object, URDFObject)
            or isinstance(physical_object, ObjectMultiplexer)
        ):
            raise ValueError("loading non-articulated object with load_articulated_object_in_renderer function")

        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []
        color = [0, 0, 0]
        for link_id in list(range(p.getNumJoints(object_pb_id))) + [-1]:
            if link_id == -1:
                link_name = p.getBodyInfo(object_pb_id)[0].decode("utf-8")
            else:
                link_name = p.getJointInfo(object_pb_id, link_id)[12].decode("utf-8")

            collision_shapes = p.getCollisionShapeData(object_pb_id, link_id)
            collision_shapes = [item for item in collision_shapes if item[2] == p.GEOM_MESH]
            # a link can have multiple collision meshes due to boxification,
            # and we want to query the original collision mesh for information

            if len(collision_shapes) == 0:
                continue
            else:
                _, _, type, dimensions, filename, rel_pos, rel_orn = collision_shapes[0]

            filenames = link_name_to_vm[link_name]
            for filename in filenames:
                overwrite_material = None
                if visual_mesh_to_material is not None and filename in visual_mesh_to_material:
                    overwrite_material = visual_mesh_to_material[filename]

                if (
                    filename,
                    tuple(dimensions),
                    tuple(rel_pos),
                    tuple(rel_orn),
                ) not in self.visual_objects.keys() or overwrite_material is not None:
                    # if the object has an overwrite material, always create a
                    # new visual object even if the same visual shape exsits
                    self.renderer.load_object(
                        filename,
                        transform_orn=rel_orn,
                        transform_pos=rel_pos,
                        input_kd=color[:3],
                        scale=np.array(dimensions),
                        overwrite_material=overwrite_material,
                    )
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))] = (
                        len(self.renderer.visual_objects) - 1
                    )
                visual_objects.append(
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))]
                )
                link_ids.append(link_id)

                if link_id == -1:
                    pos, orn = p.getBasePositionAndOrientation(object_pb_id)
                else:
                    pos, orn = p.getLinkState(object_pb_id, link_id)[:2]
                poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
                poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        self.renderer.add_instance_group(
            object_ids=visual_objects,
            link_ids=link_ids,
            pybullet_uuid=object_pb_id,
            class_id=class_id,
            poses_trans=poses_trans,
            poses_rot=poses_rot,
            dynamic=True,
            robot=None,
            use_pbr=use_pbr,
            use_pbr_mapping=use_pbr_mapping,
            shadow_caster=shadow_caster,
        )

        if physical_object is not None:
            physical_object.renderer_instances.append(self.renderer.instances[-1])

    def import_non_colliding_objects(self, objects, existing_objects=[], min_distance=0.5):
        """
        Loads objects into the scene such that they don't collide with existing objects.

        :param objects: A dictionary with objects, from a scene loaded with a particular URDF
        :param existing_objects: A list of objects that needs to be kept min_distance away when loading the new objects
        :param min_distance: A minimum distance to require for objects to load
        """
        state_id = p.saveState()
        objects_to_add = []
        for obj_name in objects:
            obj = objects[obj_name]

            # Do not allow duplicate object categories
            if obj.category in self.scene.objects_by_category:
                continue

            add = True
            body_ids = []

            # Filter based on the minimum distance to any existing object
            for idx in range(len(obj.urdf_paths)):
                body_id = p.loadURDF(obj.urdf_paths[idx])
                body_ids.append(body_id)
                transformation = obj.poses[idx]
                pos = transformation[0:3, 3]
                orn = np.array(quatXYZWFromRotMat(transformation[0:3, 0:3]))
                dynamics_info = p.getDynamicsInfo(body_id, -1)
                inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
                pos, orn = p.multiplyTransforms(pos, orn, inertial_pos, inertial_orn)
                pos = list(pos)
                min_distance_to_existing_object = None
                for existing_object in existing_objects:
                    # If a sliced obj is an existing_object, get_position will not work
                    if isinstance(existing_object, ObjectMultiplexer) and isinstance(
                        existing_object.current_selection(), ObjectGrouper
                    ):
                        obj_pos = np.array([obj.get_position() for obj in existing_object.objects]).mean(axis=0)
                    else:
                        obj_pos = existing_object.get_position()
                    distance = np.linalg.norm(np.array(pos) - np.array(obj_pos))
                    if min_distance_to_existing_object is None or min_distance_to_existing_object > distance:
                        min_distance_to_existing_object = distance

                if min_distance_to_existing_object < min_distance:
                    add = False
                    break

                pos[2] += 0.01  # slighly above to not touch furniture
                p.resetBasePositionAndOrientation(body_id, pos, orn)

            # Filter based on collisions with any existing object
            if add:
                p.stepSimulation()

                for body_id in body_ids:
                    in_collision = len(p.getContactPoints(body_id)) > 0
                    if in_collision:
                        add = False
                        break

            if add:
                objects_to_add.append(obj)

            for body_id in body_ids:
                p.removeBody(body_id)

            p.restoreState(state_id)

        p.removeState(state_id)

        for obj in objects_to_add:
            self.import_object(obj)

    @load_without_pybullet_vis
    def import_robot(self, robot, class_id=SemanticClass.ROBOTS):
        """
        Import a robot into the simulator

        :param robot: Robot
        :param class_id: Class id for rendering semantic segmentation
        :return: pybullet id
        """
        assert isinstance(robot, BaseRobot), "import_robot can only be called with BaseRobot"
        ids = robot.load()
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []
        self.robots.append(robot)

        for shape in p.getVisualShapeData(ids[0]):
            id, link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[:8]
            dynamics_info = p.getDynamicsInfo(id, link_id)
            inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
            rel_pos, rel_orn = p.multiplyTransforms(*p.invertTransform(inertial_pos, inertial_orn), rel_pos, rel_orn)
            # visual meshes frame are transformed from the urdfLinkFrame as origin to comLinkFrame as origin

            if type == p.GEOM_MESH:
                filename = filename.decode("utf-8")
                if (filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn)) not in self.visual_objects.keys():
                    self.renderer.load_object(
                        filename,
                        transform_orn=rel_orn,
                        transform_pos=rel_pos,
                        input_kd=color[:3],
                        scale=np.array(dimensions),
                    )
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))] = (
                        len(self.renderer.visual_objects) - 1
                    )
                visual_objects.append(
                    self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))]
                )
                link_ids.append(link_id)
            elif type == p.GEOM_SPHERE:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/sphere8.obj")
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5],
                )
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/cube.obj")
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=[dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]],
                )
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/cube.obj")
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=np.array(dimensions),
                )
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)

            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(id)
            else:
                pos, orn = p.getLinkState(id, link_id)[:2]
            poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        self.renderer.add_robot(
            object_ids=visual_objects,
            link_ids=link_ids,
            pybullet_uuid=ids[0],
            class_id=class_id,
            poses_rot=poses_rot,
            poses_trans=poses_trans,
            dynamic=True,
            robot=robot,
        )

        return ids

    def add_normal_text(
        self,
        text_data="PLACEHOLDER: PLEASE REPLACE!",
        font_name="OpenSans",
        font_style="Regular",
        font_size=48,
        color=[0, 0, 0],
        pos=[0, 100],
        size=[20, 20],
        scale=1.0,
        background_color=None,
    ):
        """
        Creates a Text object to be rendered to a non-VR screen. Returns the text object to the caller,
        so various settings can be changed - eg. text content, position, scale, etc.
        :param text_data: starting text to display (can be changed at a later time by set_text)
        :param font_name: name of font to render - same as font folder in iGibson assets
        :param font_style: style of font - one of [regular, italic, bold]
        :param font_size: size of font to render
        :param color: [r, g, b] color
        :param pos: [x, y] position of top-left corner of text box, in percentage across screen
        :param size: [w, h] size of text box in percentage across screen-space axes
        :param scale: scale factor for resizing text
        :param background_color: color of the background in form [r, g, b, a] - background will only appear if this is not None
        """
        # Note: For pos/size - (0,0) is bottom-left and (100, 100) is top-right
        # Calculate pixel positions for text
        pixel_pos = [int(pos[0] / 100.0 * self.renderer.width), int(pos[1] / 100.0 * self.renderer.height)]
        pixel_size = [int(size[0] / 100.0 * self.renderer.width), int(size[1] / 100.0 * self.renderer.height)]
        return self.renderer.add_text(
            text_data=text_data,
            font_name=font_name,
            font_style=font_style,
            font_size=font_size,
            color=color,
            pixel_pos=pixel_pos,
            pixel_size=pixel_size,
            scale=scale,
            background_color=background_color,
            render_to_tex=False,
        )

    def add_vr_overlay_text(
        self,
        text_data="PLACEHOLDER: PLEASE REPLACE!",
        font_name="OpenSans",
        font_style="Regular",
        font_size=48,
        color=[0, 0, 0],
        pos=[20, 80],
        size=[70, 80],
        scale=1.0,
        background_color=[1, 1, 1, 0.8],
    ):
        """
        Creates Text for use in a VR overlay. Returns the text object to the caller,
        so various settings can be changed - eg. text content, position, scale, etc.
        :param text_data: starting text to display (can be changed at a later time by set_text)
        :param font_name: name of font to render - same as font folder in iGibson assets
        :param font_style: style of font - one of [regular, italic, bold]
        :param font_size: size of font to render
        :param color: [r, g, b] color
        :param pos: [x, y] position of top-left corner of text box, in percentage across screen
        :param size: [w, h] size of text box in percentage across screen-space axes
        :param scale: scale factor for resizing text
        :param background_color: color of the background in form [r, g, b, a] - default is semi-transparent white so text is easy to read in VR
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")
        if not self.vr_overlay_initialized:
            # This function automatically creates a VR text overlay the first time text is added
            self.renderer.gen_vr_hud()
            self.vr_overlay_initialized = True

        # Note: For pos/size - (0,0) is bottom-left and (100, 100) is top-right
        # Calculate pixel positions for text
        pixel_pos = [int(pos[0] / 100.0 * self.renderer.width), int(pos[1] / 100.0 * self.renderer.height)]
        pixel_size = [int(size[0] / 100.0 * self.renderer.width), int(size[1] / 100.0 * self.renderer.height)]
        return self.renderer.add_text(
            text_data=text_data,
            font_name=font_name,
            font_style=font_style,
            font_size=font_size,
            color=color,
            pixel_pos=pixel_pos,
            pixel_size=pixel_size,
            scale=scale,
            background_color=background_color,
            render_to_tex=True,
        )

    def add_overlay_image(self, image_fpath, width=1, pos=[0, 0, -1]):
        """
        Add an image with a given file path to the VR overlay. This image will be displayed
        in addition to any text that the users wishes to display. This function returns a handle
        to the VrStaticImageOverlay, so the user can display/hide it at will.
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")
        return self.renderer.gen_static_overlay(image_fpath, width=width, pos=pos)

    def set_hud_show_state(self, show_state):
        """
        Shows/hides the main VR HUD.
        :param show_state: whether to show HUD or not
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")
        self.renderer.vr_hud.set_overlay_show_state(show_state)

    def get_hud_show_state(self):
        """
        Returns the show state of the main VR HUD.
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")
        return self.renderer.vr_hud.get_overlay_show_state()

    def _non_physics_step(self):
        """
        Complete any non-physics steps such as state updates.
        """
        # Step all of the particle systems.
        for particle_system in self.particle_systems:
            particle_system.update(self)

        # Step the object states in global topological order.
        for state_type in self.object_state_types:
            for obj in self.scene.get_objects_with_state(state_type):
                obj.states[state_type].update()

        # Step the object procedural materials based on the updated object states
        for obj in self.scene.get_objects():
            if hasattr(obj, "procedural_material") and obj.procedural_material is not None:
                obj.procedural_material.update()

    def step_vr(self, print_stats=False):
        """
        Step the simulation when using VR. Order of function calls:
        1) Simulate physics
        2) Render frame
        3) Submit rendered frame to VR compositor
        4) Update VR data for use in the next frame
        """
        assert (
            self.scene is not None
        ), "A scene must be imported before running the simulator. Use EmptyScene for an empty scene."

        # Calculate time outside of step
        outside_step_dur = 0
        if self.frame_end_time is not None:
            outside_step_dur = time.perf_counter() - self.frame_end_time
        # Simulate Physics in PyBullet
        physics_start_time = time.perf_counter()
        for _ in range(self.physics_timestep_num):
            p.stepSimulation()
        physics_dur = time.perf_counter() - physics_start_time

        non_physics_start_time = time.perf_counter()
        self._non_physics_step()
        non_physics_dur = time.perf_counter() - non_physics_start_time

        # Sync PyBullet bodies to renderer and then render to Viewer
        render_start_time = time.perf_counter()
        self.sync()
        render_dur = time.perf_counter() - render_start_time

        # Sleep until last possible Vsync
        pre_sleep_dur = outside_step_dur + physics_dur + non_physics_dur + render_dur
        sleep_start_time = time.perf_counter()
        if pre_sleep_dur < self.non_block_frame_time:
            sleep(self.non_block_frame_time - pre_sleep_dur)
        sleep_dur = time.perf_counter() - sleep_start_time

        # Update VR compositor and VR data
        vr_system_start = time.perf_counter()
        # First sync VR compositor - this is where Oculus blocks (as opposed to Vive, which blocks in update_vr_data)
        self.sync_vr_compositor()
        # Note: this should only be called once per frame - use get_vr_events to read the event data list in
        # subsequent read operations
        self.poll_vr_events()
        # This is necessary to fix the eye tracking value for the current frame, since it is multi-threaded
        self.fix_eye_tracking_value()
        # Move user to their starting location
        self.perform_vr_start_pos_move()
        # Update VR data and wait until 3ms before the next vsync
        self.renderer.update_vr_data()
        # Update VR system data - eg. offsets, haptics, etc.
        self.vr_system_update()
        vr_system_dur = time.perf_counter() - vr_system_start

        # Calculate final frame duration
        # Make sure it is non-zero for FPS calculation (set to max of 1000 if so)
        frame_dur = max(1e-3, pre_sleep_dur + sleep_dur + vr_system_dur)

        # Set variables for data saving and replay
        self.last_physics_timestep = physics_dur
        self.last_render_timestep = render_dur
        self.last_frame_dur = frame_dur

        if print_stats:
            print("Frame number {} statistics (ms)".format(self.frame_count))
            print("Total out-of-step duration: {}".format(outside_step_dur * 1000))
            print("Total physics duration: {}".format(physics_dur * 1000))
            print("Total non-physics duration: {}".format(non_physics_dur * 1000))
            print("Total render duration: {}".format(render_dur * 1000))
            print("Total sleep duration: {}".format(sleep_dur * 1000))
            print("Total VR system duration: {}".format(vr_system_dur * 1000))
            print("Total frame duration: {} and fps: {}".format(frame_dur * 1000, 1 / frame_dur))
            print(
                "Realtime factor: {}".format(round((self.physics_timestep_num * self.physics_timestep) / frame_dur, 3))
            )
            print("-------------------------")

        self.frame_count += 1
        self.frame_end_time = time.perf_counter()

    def step(self, print_stats=False):
        """
        Step the simulation at self.render_timestep and update positions in renderer
        """
        # Call separate step function for VR
        if self.can_access_vr_context:
            self.step_vr(print_stats=print_stats)
            return

        for _ in range(self.physics_timestep_num):
            p.stepSimulation()

        self._non_physics_step()
        self.sync()
        self.frame_count += 1

    def sync(self, force_sync=False):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        self.body_links_awake = 0
        for instance in self.renderer.instances:
            if instance.dynamic:
                self.body_links_awake += self.update_position(instance, force_sync=force_sync)
        if (self.use_ig_renderer or self.use_vr_renderer or self.use_simple_viewer) and self.viewer is not None:
            self.viewer.update()
        if self.first_sync:
            self.first_sync = False

    def vr_system_update(self):
        """
        Updates the VR system for a single frame. This includes moving the vr offset,
        adjusting the user's height based on button input, and triggering haptics.
        """
        # Update VR offset using appropriate controller
        if self.vr_settings.touchpad_movement:
            vr_offset_device = "{}_controller".format(self.vr_settings.movement_controller)
            is_valid, _, _ = self.get_data_for_vr_device(vr_offset_device)
            if is_valid:
                _, touch_x, touch_y = self.get_button_data_for_controller(vr_offset_device)
                new_offset = calc_offset(
                    self, touch_x, touch_y, self.vr_settings.movement_speed, self.vr_settings.relative_movement_device
                )
                self.set_vr_offset(new_offset)

        # Adjust user height based on y-axis (vertical direction) touchpad input
        vr_height_device = "left_controller" if self.vr_settings.movement_controller == "right" else "right_controller"
        is_height_valid, _, _ = self.get_data_for_vr_device(vr_height_device)
        if is_height_valid:
            curr_offset = self.get_vr_offset()
            hmd_height = self.get_hmd_world_pos()[2]
            _, _, height_y = self.get_button_data_for_controller(vr_height_device)
            if height_y < -0.7:
                vr_z_offset = -0.01
                if hmd_height + curr_offset[2] + vr_z_offset >= self.vr_settings.height_bounds[0]:
                    self.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])
            elif height_y > 0.7:
                vr_z_offset = 0.01
                if hmd_height + curr_offset[2] + vr_z_offset <= self.vr_settings.height_bounds[1]:
                    self.set_vr_offset([curr_offset[0], curr_offset[1], curr_offset[2] + vr_z_offset])

        # Update haptics for body and hands
        if self.main_vr_robot:
            vr_body_id = self.main_vr_robot.parts["body"].body_id
            vr_hands = [
                ("left_controller", self.main_vr_robot.parts["left_hand"]),
                ("right_controller", self.main_vr_robot.parts["right_hand"]),
            ]

            # Check for body haptics
            wall_ids = self.get_category_ids("walls")
            for c_info in p.getContactPoints(vr_body_id):
                if wall_ids and (c_info[1] in wall_ids or c_info[2] in wall_ids):
                    for controller in ["left_controller", "right_controller"]:
                        is_valid, _, _ = self.get_data_for_vr_device(controller)
                        if is_valid:
                            # Use 90% strength for body to warn user of collision with wall
                            self.trigger_haptic_pulse(controller, 0.9)

            # Check for hand haptics
            for hand_device, hand_obj in vr_hands:
                is_valid, _, _ = self.get_data_for_vr_device(hand_device)
                if is_valid:
                    if len(p.getContactPoints(hand_obj.body_id)) > 0 or (
                        hasattr(hand_obj, "object_in_hand") and hand_obj.object_in_hand
                    ):
                        # Only use 30% strength for normal collisions, to help add realism to the experience
                        self.trigger_haptic_pulse(hand_device, 0.3)

    def register_main_vr_robot(self, vr_robot):
        """
        Register the robot representing the VR user.
        """
        self.main_vr_robot = vr_robot

    def import_behavior_robot(self, bvr_robot):
        """
        Import registered behavior robot into the simulator.
        """
        for part_name, part_obj in bvr_robot.parts.items():
            self.import_object(part_obj, use_pbr=False, use_pbr_mapping=False, shadow_caster=True)
            if bvr_robot.use_ghost_hands and part_name in ["left_hand", "right_hand"]:
                # Ghost hands don't cast shadows
                self.import_object(part_obj.ghost_hand, use_pbr=False, use_pbr_mapping=False, shadow_caster=False)
            if part_name == "eye":
                # BREye doesn't cast shadows either
                self.import_object(
                    part_obj.head_visual_marker, use_pbr=False, use_pbr_mapping=False, shadow_caster=False
                )

    def gen_vr_data(self):
        """
        Generates a VrData object containing all of the data required to describe the VR system in the current frame.
        This data is used to power the BehaviorRobot each frame.
        """
        if not self.can_access_vr_context:
            raise RuntimeError("Unable to get VR data for current frame since VR system is not being used!")

        v = dict()
        for device in VR_DEVICES:
            is_valid, trans, rot = self.get_data_for_vr_device(device)
            device_data = [is_valid, trans.tolist(), rot.tolist()]
            device_data.extend(self.get_device_coordinate_system(device))
            v[device] = device_data
            if device in VR_CONTROLLERS:
                v["{}_button".format(device)] = self.get_button_data_for_controller(device)

        # Store final rotations of hands, with model rotation applied
        for hand in ["right", "left"]:
            # Base rotation quaternion
            base_rot = self.main_vr_robot.parts["{}_hand".format(hand)].base_rot
            # Raw rotation of controller
            controller_rot = v["{}_controller".format(hand)][2]
            # Use dummy translation to calculation final rotation
            final_rot = p.multiplyTransforms([0, 0, 0], controller_rot, [0, 0, 0], base_rot)[1]
            v["{}_controller".format(hand)].append(final_rot)

        is_valid, torso_trans, torso_rot = self.get_data_for_vr_tracker(self.vr_settings.torso_tracker_serial)
        v["torso_tracker"] = [is_valid, torso_trans, torso_rot]
        v["eye_data"] = self.get_eye_tracking_data()
        v["event_data"] = self.get_vr_events()
        reset_actions = []
        for controller in VR_CONTROLLERS:
            reset_actions.append(self.query_vr_event(controller, "reset_agent"))
        v["reset_actions"] = reset_actions
        v["vr_positions"] = [self.get_vr_pos().tolist(), list(self.get_vr_offset())]

        return VrData(v)

    def gen_vr_robot_action(self):
        """
        Generates an action for the BehaviorRobot to perform based on VrData collected this frame.

        Action space (all non-normalized values that will be clipped if they are too large)
        * See BehaviorRobot.py for details on the clipping thresholds for
        Body:
        - 6DOF pose delta - relative to body frame from previous frame
        Eye:
        - 6DOF pose delta - relative to body frame (where the body will be after applying this frame's action)
        Left hand, right hand (in that order):
        - 6DOF pose delta - relative to body frame (same as above)
        - Trigger fraction delta
        - Action reset value

        Total size: 28
        """
        # Actions are stored as 1D numpy array
        action = np.zeros((28,))

        # Get VrData for the current frame
        v = self.gen_vr_data()

        # Update body action space
        hmd_is_valid, hmd_pos, hmd_orn, hmd_r = v.query("hmd")[:4]
        torso_is_valid, torso_pos, torso_orn = v.query("torso_tracker")
        vr_body = self.main_vr_robot.parts["body"]
        prev_body_pos, prev_body_orn = vr_body.get_position_orientation()
        inv_prev_body_pos, inv_prev_body_orn = p.invertTransform(prev_body_pos, prev_body_orn)

        if self.vr_settings.using_tracked_body:
            if torso_is_valid:
                des_body_pos, des_body_orn = torso_pos, torso_orn
            else:
                des_body_pos, des_body_orn = prev_body_pos, prev_body_orn
        else:
            if hmd_is_valid:
                des_body_pos, des_body_orn = hmd_pos, p.getQuaternionFromEuler([0, 0, calc_z_rot_from_right(hmd_r)])
            else:
                des_body_pos, des_body_orn = prev_body_pos, prev_body_orn

        body_delta_pos, body_delta_orn = p.multiplyTransforms(
            inv_prev_body_pos, inv_prev_body_orn, des_body_pos, des_body_orn
        )
        action[:3] = np.array(body_delta_pos)
        action[3:6] = np.array(p.getEulerFromQuaternion(body_delta_orn))

        # Get new body position so we can calculate correct relative transforms for other VR objects
        clipped_body_delta_pos, clipped_body_delta_orn = vr_body.clip_delta_pos_orn(action[:3], action[3:6])
        clipped_body_delta_orn = p.getQuaternionFromEuler(clipped_body_delta_orn)
        new_body_pos, new_body_orn = p.multiplyTransforms(
            prev_body_pos, prev_body_orn, clipped_body_delta_pos, clipped_body_delta_orn
        )
        # Also calculate its inverse for further local transform calculations
        inv_new_body_pos, inv_new_body_orn = p.invertTransform(new_body_pos, new_body_orn)

        # Update action space for other VR objects
        body_relative_parts = ["right", "left", "eye"]
        for part_name in body_relative_parts:
            vr_part = (
                self.main_vr_robot.parts[part_name]
                if part_name == "eye"
                else self.main_vr_robot.parts["{}_hand".format(part_name)]
            )

            # Process local transform adjustments
            prev_world_pos, prev_world_orn = vr_part.get_position_orientation()
            prev_local_pos, prev_local_orn = vr_part.local_pos, vr_part.local_orn
            _, inv_prev_local_orn = p.invertTransform(prev_local_pos, prev_local_orn)
            if part_name == "eye":
                valid, world_pos, world_orn = hmd_is_valid, hmd_pos, hmd_orn
            else:
                valid, world_pos, _ = v.query("{}_controller".format(part_name))[:3]
                # Need rotation of the model so it will appear aligned with the physical controller in VR
                world_orn = v.query("{}_controller".format(part_name))[6]

            # Keep in same world position as last frame if controller/tracker data is not valid
            if not valid:
                world_pos, world_orn = prev_world_pos, prev_world_orn

            # Get desired local position and orientation transforms
            des_local_pos, des_local_orn = p.multiplyTransforms(
                inv_new_body_pos, inv_new_body_orn, world_pos, world_orn
            )

            # Get the delta local orientation in the reference frame of the body
            _, delta_local_orn = p.multiplyTransforms(
                [0, 0, 0],
                des_local_orn,
                [0, 0, 0],
                inv_prev_local_orn,
            )
            delta_local_orn = p.getEulerFromQuaternion(delta_local_orn)

            # Get the delta local position in the reference frame of the body
            delta_local_pos = np.array(des_local_pos) - np.array(prev_local_pos)

            if part_name == "eye":
                action[6:9] = np.array(delta_local_pos)
                action[9:12] = np.array(delta_local_orn)
            elif part_name == "left":
                action[12:15] = np.array(delta_local_pos)
                action[15:18] = np.array(delta_local_orn)
            else:
                action[20:23] = np.array(delta_local_pos)
                action[23:26] = np.array(delta_local_orn)

            # Process trigger fraction and reset for controllers
            if part_name in ["right", "left"]:
                prev_trig_frac = vr_part.trigger_fraction
                if valid:
                    trig_frac = v.query("{}_controller_button".format(part_name))[0]
                    delta_trig_frac = trig_frac - prev_trig_frac
                else:
                    delta_trig_frac = 0.0
                if part_name == "left":
                    action[18] = delta_trig_frac
                else:
                    action[26] = delta_trig_frac
                # If we reset, action is 1, otherwise 0
                reset_action = v.query("reset_actions")[0] if part_name == "left" else v.query("reset_actions")[1]
                reset_action_val = 1.0 if reset_action else 0.0
                if part_name == "left":
                    action[19] = reset_action_val
                else:
                    action[27] = reset_action_val

        return action

    def sync_vr_compositor(self):
        """
        Sync VR compositor.
        """
        self.renderer.vr_compositor_update()

    def perform_vr_start_pos_move(self):
        """
        Sets the VR position on the first step iteration where the hmd tracking is valid. Not to be confused
        with self.set_vr_start_pos, which simply records the desired start position before the simulator starts running.
        """
        # Update VR start position if it is not None and the hmd is valid
        # This will keep checking until we can successfully set the start position
        if self.vr_start_pos:
            hmd_is_valid, _, _, _ = self.renderer.vrsys.getDataForVRDevice("hmd")
            if hmd_is_valid:
                offset_to_start = np.array(self.vr_start_pos) - self.get_hmd_world_pos()
                if self.vr_height_offset is not None:
                    offset_to_start[2] = self.vr_height_offset
                self.set_vr_offset(offset_to_start)
                self.vr_start_pos = None

    def fix_eye_tracking_value(self):
        """
        Calculates and fixes eye tracking data to its value during step(). This is necessary, since multiple
        calls to get eye tracking data return different results, due to the SRAnipal multithreaded loop that
        runs in parallel to the iGibson main thread
        """
        self.eye_tracking_data = self.renderer.vrsys.getEyeTrackingData()

    def gen_assisted_grasping_categories(self):
        """
        Generates list of categories that can be grasped using assisted grasping,
        using labels provided in average category specs file.
        """
        avg_category_spec = get_ig_avg_category_specs()
        for k, v in avg_category_spec.items():
            if v["enable_ag"]:
                self.assist_grasp_category_allow_list.add(k)

    def can_assisted_grasp(self, body_id, c_link):
        """
        Checks to see if an object with the given body_id can be grasped. This is done
        by checking its category to see if is in the allowlist.
        """
        if (
            not hasattr(self.scene, "objects_by_id")
            or body_id not in self.scene.objects_by_id
            or not hasattr(self.scene.objects_by_id[body_id], "category")
            or self.scene.objects_by_id[body_id].category == "object"
        ):
            mass = p.getDynamicsInfo(body_id, c_link)[0]
            return mass <= self.assist_grasp_mass_thresh
        else:
            return self.scene.objects_by_id[body_id].category in self.assist_grasp_category_allow_list

    def poll_vr_events(self):
        """
        Returns VR event data as list of lists.
        List is empty if all events are invalid. Components of a single event:
        controller: 0 (left_controller), 1 (right_controller)
        button_idx: any valid idx in EVRButtonId enum in openvr.h header file
        press: 0 (unpress), 1 (press)
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        self.vr_event_data = self.renderer.vrsys.pollVREvents()
        # Enforce store_first_button_press_per_frame option, if user has enabled it
        if self.vr_settings.store_only_first_event_per_button:
            temp_event_data = []
            # Make sure we only store the first (button, press) combo of each type
            event_set = set()
            for ev_data in self.vr_event_data:
                controller, button_idx, _ = ev_data
                key = (controller, button_idx)
                if key not in event_set:
                    temp_event_data.append(ev_data)
                    event_set.add(key)
            self.vr_event_data = temp_event_data[:]

        return self.vr_event_data

    def get_vr_events(self):
        """
        Returns the VR events processed by the simulator
        """
        return self.vr_event_data

    def get_button_for_action(self, action):
        """
        Returns (button, state) tuple corresponding to an action
        :param action: an action name listed in "action_button_map" dictionary for the current device in the vr_config.yml
        """
        return (
            None
            if action not in self.vr_settings.action_button_map
            else tuple(self.vr_settings.action_button_map[action])
        )

    def query_vr_event(self, controller, action):
        """
        Queries system for a VR event, and returns true if that event happened this frame
        :param controller: device to query for - can be left_controller or right_controller
        :param action: an action name listed in "action_button_map" dictionary for the current device in the vr_config.yml
        """
        # Return false if any of input parameters are invalid
        if (
            controller not in ["left_controller", "right_controller"]
            or action not in self.vr_settings.action_button_map.keys()
        ):
            return False

        # Search through event list to try to find desired event
        controller_id = 0 if controller == "left_controller" else 1
        button_idx, press_id = self.vr_settings.action_button_map[action]
        for ev_data in self.vr_event_data:
            if controller_id == ev_data[0] and button_idx == ev_data[1] and press_id == ev_data[2]:
                return True

        # Return false if event was not found this frame
        return False

    def get_data_for_vr_device(self, device_name):
        """
        Call this after step - returns all VR device data for a specific device
        Returns is_valid (indicating validity of data), translation and rotation in Gibson world space
        :param device_name: can be hmd, left_controller or right_controller
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        # Use fourth variable in list to get actual hmd position in space
        is_valid, translation, rotation, _ = self.renderer.vrsys.getDataForVRDevice(device_name)
        if not is_valid:
            translation = np.array([0, 0, 0])
            rotation = np.array([0, 0, 0, 1])
        return [is_valid, translation, rotation]

    def get_data_for_vr_tracker(self, tracker_serial_number):
        """
        Returns the data for a tracker with a specific serial number. This number can be found
        by looking in the SteamVR device information.
        :param tracker_serial_number: the serial number of the tracker
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        if not tracker_serial_number:
            return [False, [0, 0, 0], [0, 0, 0, 0]]

        tracker_data = self.renderer.vrsys.getDataForVRTracker(tracker_serial_number)
        # Set is_valid to false, and assume the user will check for invalid data
        if not tracker_data:
            return [False, np.array([0, 0, 0]), np.array([0, 0, 0, 1])]

        is_valid, translation, rotation = tracker_data
        return [is_valid, translation, rotation]

    def get_hmd_world_pos(self):
        """
        Get world position of HMD without offset
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        _, _, _, hmd_world_pos = self.renderer.vrsys.getDataForVRDevice("hmd")
        return hmd_world_pos

    def get_button_data_for_controller(self, controller_name):
        """
        Call this after getDataForVRDevice - returns analog data for a specific controller
        Returns trigger_fraction, touchpad finger position x, touchpad finger position y
        Data is only valid if isValid is true from previous call to getDataForVRDevice
        Trigger data: 1 (closed) <------> 0 (open)
        Analog data: X: -1 (left) <-----> 1 (right) and Y: -1 (bottom) <------> 1 (top)
        :param controller_name: one of left_controller or right_controller
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        # Test for validity when acquiring button data
        if self.get_data_for_vr_device(controller_name)[0]:
            trigger_fraction, touch_x, touch_y = self.renderer.vrsys.getButtonDataForController(controller_name)
        else:
            trigger_fraction, touch_x, touch_y = 0.0, 0.0, 0.0
        return [trigger_fraction, touch_x, touch_y]

    def get_scroll_input(self):
        """
        Gets scroll input. This uses the non-movement-controller, and determines whether
        the user wants to scroll by testing if they have pressed the touchpad, while keeping
        their finger on the left/right of the pad. Return True for up and False for down (-1 for no scroll)
        """
        mov_controller = self.vr_settings.movement_controller
        other_controller = "right" if mov_controller == "left" else "left"
        other_controller = "{}_controller".format(other_controller)
        # Data indicating whether user has pressed top or bottom of the touchpad
        _, touch_x, _ = self.renderer.vrsys.getButtonDataForController(other_controller)
        # Detect no touch in extreme regions of x axis
        if touch_x > 0.7 and touch_x <= 1.0:
            return 1
        elif touch_x < -0.7 and touch_x >= -1.0:
            return 0
        else:
            return -1

    def get_eye_tracking_data(self):
        """
        Returns eye tracking data as list of lists. Order: is_valid, gaze origin, gaze direction, gaze point,
        left pupil diameter, right pupil diameter (both in millimeters)
        Call after getDataForVRDevice, to guarantee that latest HMD transform has been acquired
        """
        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = self.eye_tracking_data
        # Set other values to 0 to avoid very small/large floating point numbers
        if not is_valid:
            return [False, [0, 0, 0], [0, 0, 0], 0, 0]
        else:
            return [is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter]

    def set_vr_start_pos(self, start_pos=None, vr_height_offset=None):
        """
        Sets the starting position of the VR system in iGibson space
        :param start_pos: position to start VR system at
        :param vr_height_offset: starting height offset. If None, uses absolute height from start_pos
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        # The VR headset will actually be set to this position during the first frame.
        # This is because we need to know where the headset is in space when it is first picked
        # up to set the initial offset correctly.
        self.vr_start_pos = start_pos
        # This value can be set to specify a height offset instead of an absolute height.
        # We might want to adjust the height of the camera based on the height of the person using VR,
        # but still offset this height. When this option is not None it offsets the height by the amount
        # specified instead of overwriting the VR system height output.
        self.vr_height_offset = vr_height_offset

    def set_vr_pos(self, pos=None, keep_height=False):
        """
        Sets the world position of the VR system in iGibson space
        :param pos: position to set VR system to
        :param keep_height: whether the current VR height should be kept
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        offset_to_pos = np.array(pos) - self.get_hmd_world_pos()
        if keep_height:
            curr_offset_z = self.get_vr_offset()[2]
            self.set_vr_offset([offset_to_pos[0], offset_to_pos[1], curr_offset_z])
        else:
            self.set_vr_offset(offset_to_pos)

    def get_vr_pos(self):
        """
        Gets the world position of the VR system in iGibson space.
        """
        return self.get_hmd_world_pos() + np.array(self.get_vr_offset())

    def set_vr_offset(self, pos=None):
        """
        Sets the translational offset of the VR system (HMD, left controller, right controller) from world space coordinates.
        Can be used for many things, including adjusting height and teleportation-based movement
        :param pos: must be a list of three floats, corresponding to x, y, z in Gibson coordinate space
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        self.renderer.vrsys.setVROffset(-pos[1], pos[2], -pos[0])

    def get_vr_offset(self):
        """
        Gets the current VR offset vector in list form: x, y, z (in iGibson coordinates)
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        x, y, z = self.renderer.vrsys.getVROffset()
        return [x, y, z]

    def get_device_coordinate_system(self, device):
        """
        Gets the direction vectors representing the device's coordinate system in list form: x, y, z (in Gibson coordinates)
        List contains "right", "up" and "forward" vectors in that order
        :param device: can be one of "hmd", "left_controller" or "right_controller"
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")

        vec_list = []

        coordinate_sys = self.renderer.vrsys.getDeviceCoordinateSystem(device)
        for dir_vec in coordinate_sys:
            vec_list.append(dir_vec)

        return vec_list

    def trigger_haptic_pulse(self, device, strength):
        """
        Triggers a haptic pulse of the specified strength (0 is weakest, 1 is strongest)
        :param device: device to trigger haptic for - can be any one of [left_controller, right_controller]
        :param strength: strength of haptic pulse (0 is weakest, 1 is strongest)
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")
        assert device in ["left_controller", "right_controller"]

        self.renderer.vrsys.triggerHapticPulseForDevice(device, int(self.max_haptic_duration * strength))

    def set_hidden_state(self, obj, hide=True):
        """
        Sets the hidden state of an object to be either hidden or not hidden.
        The object passed in must inherent from Object at the top level

        Note: this function must be called after step() in the rendering loop
        Note 2: this function only works with the optimized renderer - please use the renderer hidden
        list to hide objects in the non-optimized renderer
        """
        # Find instance corresponding to this id in the renderer
        for instance in self.renderer.instances:
            if obj.body_id == instance.pybullet_uuid:
                instance.hidden = hide
                self.renderer.update_hidden_highlight_state([instance])
                return

    def set_hud_state(self, state):
        """
        Sets state of the VR HUD (heads-up-display)
        :param state: one of 'show' or 'hide'
        """
        if not self.can_access_vr_context:
            raise RuntimeError("ERROR: Trying to access VR context without enabling vr mode and use_vr in vr settings!")
        if self.renderer.vr_hud:
            self.renderer.vr_hud.set_overlay_state(state)

    def get_hidden_state(self, obj):
        """
        Returns the current hidden state of the object - hidden (True) or not hidden (False)
        """
        for instance in self.renderer.instances:
            if obj.body_id == instance.pybullet_uuid:
                return instance.hidden

    def get_category_ids(self, category_name):
        """
        Gets ids for all instances of a specific category (floors, walls, etc.) in a scene
        """
        if not hasattr(self.scene, "objects_by_id"):
            return []
        return [
            body_id
            for body_id in self.objects
            if (
                body_id in self.scene.objects_by_id.keys()
                and hasattr(self.scene.objects_by_id[body_id], "category")
                and self.scene.objects_by_id[body_id].category == category_name
            )
        ]

    def update_position(self, instance, force_sync=False):
        """
        Update position for an object or a robot in renderer.
        :param instance: Instance in the renderer
        """
        body_links_awake = 0
        if isinstance(instance, Instance):
            dynamics_info = p.getDynamicsInfo(instance.pybullet_uuid, -1)
            inertial_pos = dynamics_info[3]
            inertial_orn = dynamics_info[4]
            if len(dynamics_info) == 13 and not self.first_sync and not force_sync:
                activation_state = dynamics_info[12]
            else:
                activation_state = PyBulletSleepState.AWAKE

            if activation_state != PyBulletSleepState.AWAKE:
                return body_links_awake
            # pos and orn of the inertial frame of the base link,
            # instead of the base link frame
            pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)

            # Need to convert to the base link frame because that is
            # what our own renderer keeps track of
            # Based on pyullet docuementation:
            # urdfLinkFrame = comLinkFrame * localInertialFrame.inverse().

            instance.set_position(pos)
            instance.set_rotation(quat2rotmat(xyzw2wxyz(orn)))
            body_links_awake += 1
        elif isinstance(instance, InstanceGroup):
            for j, link_id in enumerate(instance.link_ids):
                if link_id == -1:
                    dynamics_info = p.getDynamicsInfo(instance.pybullet_uuid, -1)
                    inertial_pos = dynamics_info[3]
                    inertial_orn = dynamics_info[4]
                    if len(dynamics_info) == 13 and not self.first_sync:
                        activation_state = dynamics_info[12]
                    else:
                        activation_state = PyBulletSleepState.AWAKE

                    if activation_state != PyBulletSleepState.AWAKE:
                        continue
                    # same conversion is needed as above
                    pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)

                else:
                    dynamics_info = p.getDynamicsInfo(instance.pybullet_uuid, link_id)

                    if len(dynamics_info) == 13 and not self.first_sync:
                        activation_state = dynamics_info[12]
                    else:
                        activation_state = PyBulletSleepState.AWAKE

                    if activation_state != PyBulletSleepState.AWAKE:
                        continue

                    pos, orn = p.getLinkState(instance.pybullet_uuid, link_id)[:2]

                instance.set_position_for_part(xyz2mat(pos), j)
                instance.set_rotation_for_part(quat2rotmat(xyzw2wxyz(orn)), j)
                body_links_awake += 1
        return body_links_awake

    def isconnected(self):
        """
        :return: pybullet is alive
        """
        return p.getConnectionInfo(self.cid)["isConnected"]

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
