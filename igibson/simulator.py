import logging
import os
import platform

import numpy as np
import pybullet as p

import igibson
from igibson.object_states.factory import get_states_by_dependency_order
from igibson.objects.multi_object_wrappers import ObjectGrouper, ObjectMultiplexer
from igibson.objects.object_base import NonRobotObject
from igibson.objects.particles import Particle, ParticleSystem
from igibson.objects.visual_marker import VisualMarker
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from igibson.render.viewer import Viewer, ViewerSimple
from igibson.robots.behavior_robot import BehaviorRobot
from igibson.robots.robot_base import BaseRobot
from igibson.scenes.scene_base import Scene
from igibson.utils.assets_utils import get_ig_avg_category_specs
from igibson.utils.constants import PyBulletSleepState, SimulatorMode
from igibson.utils.mesh_util import quat2rotmat, xyz2mat, xyzw2wxyz
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
            self.use_pb_gui = False  # for mac os disable pybullet rendering
            logging.warn(
                "Simulator mode gui_interactive is not supported when `use_pb_gui` is true on macOS. Default to use_pb_gui = False."
            )

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
    def import_scene(self, scene):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: Scene object
        :return: pybullet body ids from scene.load function
        """
        assert isinstance(scene, Scene), "import_scene can only be called with Scene"
        scene.load(self)
        self.scene = scene

    @load_without_pybullet_vis
    def import_particle_system(self, particle_system):
        """
        Import an object into the simulator. Called by objects owning a particle-system, via reference to the Simulator instance.
        :param obj: ParticleSystem to load
        """

        assert isinstance(
            particle_system, ParticleSystem
        ), "import_particle_system can only be called with ParticleSystem"

        self.particle_systems.append(particle_system)
        particle_system.initialize(self)

    @load_without_pybullet_vis
    def import_object(self, obj):
        """
        Import an object into the simulator

        :param obj: Object to load
        """
        assert isinstance(obj, NonRobotObject), "import_object can only be called with NonRobotObject"

        if isinstance(obj, VisualMarker) or isinstance(obj, Particle):
            # Marker objects can be imported without a scene.
            obj.load(self)
        else:
            # Non-marker objects require a Scene to be imported.
            # Load the object in pybullet. Returns a pybullet id that we can use to load it in the renderer
            self.scene.add_object(obj, self, _is_call_from_simulator=True)

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

            restoreState(state_id)

        p.removeState(state_id)

        for obj in objects_to_add:
            self.import_object(obj)

    @load_without_pybullet_vis
    def import_robot(self, robot):
        """
        Import a robot into the simulator

        :param robot: Robot
        :param class_id: Class id for rendering semantic segmentation
        :return: pybullet id
        """
        assert isinstance(robot, (BaseRobot, BehaviorRobot)), "import_robot can only be called with Robots"
        robot.load(self)
        self.robots.append(robot)

    @load_without_pybullet_vis
    def load_object_in_renderer(
        self,
        obj,
        body_id,
        class_id,
        link_name_to_vm=None,
        visual_mesh_to_material=None,
        use_pbr=True,
        use_pbr_mapping=True,
        shadow_caster=True,
        softbody=False,
        texture_scale=1.0,
    ):
        # First, grab all the visual shapes.
        if link_name_to_vm:
            # If a manual link-name-to-visual-mesh mapping is given, use that to generate list of shapes.
            shapes = []
            for link_id in list(range(p.getNumJoints(body_id))) + [-1]:
                if link_id == -1:
                    link_name = p.getBodyInfo(body_id)[0].decode("utf-8")
                else:
                    link_name = p.getJointInfo(body_id, link_id)[12].decode("utf-8")

                collision_shapes = p.getCollisionShapeData(body_id, link_id)
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

                    shapes.append(
                        (link_id, p.GEOM_MESH, dimensions, filename, rel_pos, rel_orn, [0, 0, 0], overwrite_material)
                    )
        else:
            # Pull the visual shapes from pybullet
            shapes = []
            for shape in p.getVisualShapeData(body_id):
                link_id, type, dimensions, filename, rel_pos, rel_orn, color = shape[1:8]

                if filename:
                    filename = filename.decode("utf-8")

                # visual meshes frame are transformed from the urdfLinkFrame as origin to comLinkFrame as origin
                dynamics_info = p.getDynamicsInfo(body_id, link_id)
                inertial_pos, inertial_orn = dynamics_info[3], dynamics_info[4]
                rel_pos, rel_orn = p.multiplyTransforms(
                    *p.invertTransform(inertial_pos, inertial_orn), rel_pos, rel_orn
                )
                shapes.append((link_id, type, dimensions, filename, rel_pos, rel_orn, color, None))

        # Now that we have the visual shapes, let's add them to the renderer.
        visual_objects = []
        link_ids = []
        poses_rot = []
        poses_trans = []
        for shape in shapes:
            link_id, type, dimensions, filename, rel_pos, rel_orn, color, overwrite_material = shape

            # Specify a filename if our object is not a mesh
            if type == p.GEOM_SPHERE:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/sphere8.obj")
                dimensions = [dimensions[0] / 0.5, dimensions[0] / 0.5, dimensions[0] / 0.5]
            elif type == p.GEOM_CAPSULE or type == p.GEOM_CYLINDER:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/cube.obj")
                dimensions = [dimensions[1] / 0.5, dimensions[1] / 0.5, dimensions[0]]
            elif type == p.GEOM_BOX:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/cube.obj")
            elif type == p.GEOM_PLANE:
                filename = os.path.join(igibson.assets_path, "models/mjcf_primitives/cube.obj")
                dimensions = [100, 100, 0.01]

            # Load the visual object if it doesn't already exist.
            if (filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn)) not in self.visual_objects.keys():
                self.renderer.load_object(
                    filename,
                    transform_orn=rel_orn,
                    transform_pos=rel_pos,
                    input_kd=color[:3],
                    scale=np.array(dimensions),
                    texture_scale=texture_scale,
                    overwrite_material=overwrite_material,
                )
                self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))] = (
                    len(self.renderer.visual_objects) - 1
                )

            # Keep track of the objects we just loaded.
            visual_objects.append(self.visual_objects[(filename, tuple(dimensions), tuple(rel_pos), tuple(rel_orn))])
            link_ids.append(link_id)

            # Keep track of the positions.
            if link_id == -1:
                pos, orn = p.getBasePositionAndOrientation(body_id)
            else:
                pos, orn = p.getLinkState(body_id, link_id)[:2]
            poses_rot.append(np.ascontiguousarray(quat2rotmat(xyzw2wxyz(orn))))
            poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

        # Finally, create and add the instance group for this object.
        self.renderer.add_instance_group(
            object_ids=visual_objects,
            link_ids=link_ids,
            pybullet_uuid=body_id,
            ig_object=obj,
            class_id=class_id,
            poses_trans=poses_trans,
            poses_rot=poses_rot,
            softbody=softbody,
            dynamic=True,
            use_pbr=use_pbr,
            use_pbr_mapping=use_pbr_mapping,
            shadow_caster=shadow_caster,
        )

        # Add the instance onto the object as well.
        # TODO: Remove condition and/or migrate to owning classes.
        if hasattr(obj, "renderer_instances"):
            obj.renderer_instances.append(self.renderer.instances[-1])

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

    @staticmethod
    def update_position(instance, force_sync=False):
        """
        Update position for an object or a robot in renderer.
        :param instance: Instance in the renderer
        """
        body_links_awake = 0
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
