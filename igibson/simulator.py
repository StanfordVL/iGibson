import logging
import os
import platform

import numpy as np
import pybullet as p

import igibson
from igibson.object_states.factory import get_states_by_dependency_order
from igibson.objects.articulated_object import ArticulatedObject, URDFObject
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.objects.object_base import NonRobotObject
from igibson.objects.particles import Particle, ParticleSystem
from igibson.objects.stateful_object import StatefulObject
from igibson.objects.visual_marker import VisualMarker
from igibson.render.mesh_renderer.instances import Instance, InstanceGroup
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.viewer import Viewer, ViewerSimple

# REMOVE
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.robots.robot_base import BaseRobot
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.scenes.scene_base import Scene
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.utils.constants import PyBulletSleepState, SemanticClass, SimulatorMode
from igibson.utils.mesh_util import quat2rotmat, xyz2mat, xyzw2wxyz
from igibson.utils.semantics_utils import get_class_name_to_class_id
from igibson.utils.utils import quatXYZWFromRotMat, restoreState


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
        mode="gui_interactive",
        image_width=128,
        image_height=128,
        vertical_fov=90,
        device_idx=0,
        rendering_settings=MeshRendererSettings(),
        use_pb_gui=False,
    ):
        """
        :param gravity: gravity on z direction.
        :param physics_timestep: timestep of physical simulation, p.stepSimulation()
        :param render_timestep: timestep of rendering, and Simulator.step() function
        :param solver_iterations: number of solver iterations to feed into pybullet, can be reduced to increase speed.
            pybullet default value is 50.
        :param use_variable_step_num: whether to use a fixed (1) or variable physics step number
        :param mode: choose mode from gui_interactive, gui_non_interactive, headless, headless_tensor
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        disable it when you want to run multiple physics step but don't need to visualize each frame
        :param rendering_settings: settings to use for mesh renderer
        :param use_pb_gui: concurrently display the interactive pybullet gui (for debugging)
        """
        # physics simulator
        self.gravity = gravity
        self.physics_timestep = physics_timestep
        self.render_timestep = render_timestep
        self.solver_iterations = solver_iterations
        self.physics_timestep_num = self.render_timestep / self.physics_timestep
        assert self.physics_timestep_num.is_integer(), "render_timestep must be a multiple of physics_timestep"

        self.physics_timestep_num = int(self.physics_timestep_num)
        self.mode = SimulatorMode[mode.upper()]

        self.particle_systems = []
        self.objects = []

        self.image_width = image_width
        self.image_height = image_height
        self.vertical_fov = vertical_fov
        self.device_idx = device_idx
        self.rendering_settings = rendering_settings
        self.use_pb_gui = use_pb_gui

        plt = platform.system()
        if plt == "Darwin" and self.mode == SimulatorMode.GUI_INTERACTIVE and use_pb_gui:
            use_pb_gui = False  # for mac os disable pybullet rendering
            logging.warn("Simulator mode gui_interactive is not supported when `use_pb_gui` is true on macOS")

        self.frame_count = 0
        self.body_links_awake = 0
        self.viewer = None
        self.renderer = None

        self.initialize_physics_engine()
        self.initialize_renderer()

        self.robots = []

        # First sync always sync all objects (regardless of their sleeping states)
        self.first_sync = True

        # Set of categories that can be grasped by assisted grasping
        self.class_name_to_class_id = get_class_name_to_class_id()
        self.object_state_types = get_states_by_dependency_order()

        self.assist_grasp_category_allow_list = self.gen_assisted_grasping_categories()
        self.assist_grasp_mass_thresh = 10.0

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

    def disconnect(self, release_renderer=True):
        """
        Clean up the simulator
        """
        if p.getConnectionInfo(self.cid)["isConnected"]:
            # print("******************PyBullet Logging Information:")
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)
            # print("PyBullet Logging Information******************")
        if release_renderer:
            self.renderer.release()

    def reload(self):
        """
        Destroy the MeshRenderer and physics simulator and start again.
        """
        self.disconnect()
        self.initialize_physics_engine()
        self.initialize_renderer()

    def initialize_physics_engine(self):
        if self.use_pb_gui:
            self.cid = p.connect(p.GUI)
        else:
            self.cid = p.connect(p.DIRECT)

        # Needed for deterministic action replay
        # TODO(mjlbach) consider making optional and benchmark
        p.resetSimulation()
        p.setPhysicsEngineParameter(deterministicOverlappingPairs=1)
        p.setPhysicsEngineParameter(numSolverIterations=self.solver_iterations)
        p.setTimeStep(self.physics_timestep)
        p.setGravity(0, 0, -self.gravity)
        p.setPhysicsEngineParameter(enableFileCaching=0)

    def initialize_renderer(self):
        self.visual_objects = {}
        if self.mode == SimulatorMode.HEADLESS_TORCH:
            self.renderer = MeshRendererG2G(
                width=self.image_width,
                height=self.image_height,
                vertical_fov=self.vertical_fov,
                device_idx=self.device_idx,
                rendering_settings=self.rendering_settings,
                simulator=self,
            )
        elif self.mode in [SimulatorMode.GUI_INTERACTIVE, SimulatorMode.GUI_NON_INTERACTIVE, SimulatorMode.HEADLESS]:
            self.renderer = MeshRenderer(
                width=self.image_width,
                height=self.image_height,
                vertical_fov=self.vertical_fov,
                device_idx=self.device_idx,
                rendering_settings=self.rendering_settings,
                simulator=self,
            )
        else:
            raise Exception(
                "The available render modes are headless_torch, gui_interactive, gui_non_interactive, and headless"
            )

        if self.mode == SimulatorMode.GUI_NON_INTERACTIVE:
            self.viewer = ViewerSimple(renderer=self.renderer)
        elif self.mode == SimulatorMode.GUI_INTERACTIVE:
            self.viewer = Viewer(simulator=self, renderer=self.renderer)

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
            if isinstance(obj, ObjectMultiplexer):
                for sub_obj in obj._multiplexed_objects:
                    if isinstance(sub_obj, ObjectGrouper):
                        for group_sub_obj in sub_obj.objects:
                            for state in group_sub_obj.states.values():
                                state.initialize(self)
                    else:
                        for state in sub_obj.states.values():
                            state.initialize(self)
            elif isinstance(obj, StatefulObject):
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
        if not self.use_pb_gui:
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
            if isinstance(obj, ObjectMultiplexer):
                for sub_obj in obj._multiplexed_objects:
                    if isinstance(sub_obj, ObjectGrouper):
                        for group_sub_obj in sub_obj.objects:
                            for state in group_sub_obj.states.values():
                                state.initialize(self)
                    else:
                        for state in sub_obj.states.values():
                            state.initialize(self)
            elif isinstance(obj, StatefulObject):
                for state in obj.states.values():
                    state.initialize(self)

        return new_object_ids

    @load_without_pybullet_vis
    def import_particle_system(self, obj):
        """
        Import an object into the simulator. Called by objects owning a particle-system, via reference to the Simulator instance.
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
        assert isinstance(obj, NonRobotObject), "import_object can only be called with NonRobotObject"

        if isinstance(obj, VisualMarker) or isinstance(obj, Particle):
            # Marker objects can be imported without a scene.
            new_object_pb_ids = obj.load()
        else:
            # Non-marker objects require a Scene to be imported.
            # Load the object in pybullet. Returns a pybullet id that we can use to load it in the renderer
            new_object_pb_ids = self.scene.add_object(obj, _is_call_from_simulator=True)

        # If no new bodies are immediately imported into pybullet, we have no rendering steps.
        if not new_object_pb_ids:
            return None

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
        if isinstance(obj, ObjectMultiplexer):
            for sub_obj in obj._multiplexed_objects:
                if isinstance(sub_obj, ObjectGrouper):
                    for group_sub_obj in sub_obj.objects:
                        for state in group_sub_obj.states.values():
                            state.initialize(self)
                else:
                    for state in sub_obj.states.values():
                        state.initialize(self)
        elif isinstance(obj, StatefulObject):
            for state in obj.states.values():
                state.initialize(self)

        return new_object_pb_ids

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
                _, _, _, dimensions, filename, rel_pos, rel_orn = collision_shapes[0]

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
        assert ids
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

        for state in robot.states.values():
            state.initialize(self)

        return ids

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

    def step(self):
        """
        Step the simulation at self.render_timestep and update positions in renderer
        """
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
                self.body_links_awake += self.update_position(instance, force_sync=force_sync or self.first_sync)
        if self.viewer is not None:
            self.viewer.update()
        if self.first_sync:
            self.first_sync = False

    def import_behavior_robot(self, bvr_robot):
        """
        Import registered behavior robot into the simulator.
        """
        assert isinstance(bvr_robot, BehaviorRobot), "import_robot can only be called with BaseRobot"
        self.robots.append(bvr_robot)
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

    def gen_assisted_grasping_categories(self):
        """
        Generates list of categories that can be grasped using assisted grasping,
        using labels provided in average category specs file.
        """
        assisted_grasp_category_allow_list = set()
        avg_category_spec = get_ig_avg_category_specs()
        for k, v in avg_category_spec.items():
            if v["enable_ag"]:
                assisted_grasp_category_allow_list.add(k)
        return assisted_grasp_category_allow_list

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
            if obj.get_body_id() == instance.pybullet_uuid:
                instance.hidden = hide
                self.renderer.update_hidden_highlight_state([instance])
                return

    def get_hidden_state(self, obj):
        """
        Returns the current hidden state of the object - hidden (True) or not hidden (False)
        """
        for instance in self.renderer.instances:
            if obj.get_body_id() == instance.pybullet_uuid:
                return instance.hidden

    def get_category_ids(self, category_name):
        """
        Gets ids for all instances of a specific category (floors, walls, etc.) in a scene
        """
        assert self.scene is not None, "Category IDs require a loaded scene"

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

    @staticmethod
    def update_position(instance, force_sync=False):
        """
        Update position for an object or a robot in renderer.
        :param instance: Instance in the renderer
        """
        body_links_awake = 0
        if isinstance(instance, Instance):
            dynamics_info = p.getDynamicsInfo(instance.pybullet_uuid, -1)
            if len(dynamics_info) == 13 and not force_sync:
                activation_state = dynamics_info[12]
            else:
                activation_state = PyBulletSleepState.AWAKE

            if activation_state != PyBulletSleepState.AWAKE:
                return body_links_awake

            pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
            instance.set_position(pos)
            instance.set_rotation(quat2rotmat(xyzw2wxyz(orn)))
            body_links_awake += 1

        elif isinstance(instance, InstanceGroup):
            for j, link_id in enumerate(instance.link_ids):
                if link_id == -1:
                    dynamics_info = p.getDynamicsInfo(instance.pybullet_uuid, -1)
                    if len(dynamics_info) == 13 and not force_sync:
                        activation_state = dynamics_info[12]
                    else:
                        activation_state = PyBulletSleepState.AWAKE

                    if activation_state != PyBulletSleepState.AWAKE:
                        continue

                    pos, orn = p.getBasePositionAndOrientation(instance.pybullet_uuid)
                else:
                    dynamics_info = p.getDynamicsInfo(instance.pybullet_uuid, link_id)

                    if len(dynamics_info) == 13 and not force_sync:
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
