from gibson2.utils.mesh_util import quat2rotmat, xyzw2wxyz, xyz2mat
from gibson2.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer, InstanceGroup, Instance, MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
from gibson2.render.viewer import Viewer, ViewerVR
from gibson2.objects.articulated_object import ArticulatedObject, URDFObject
import pybullet as p
import gibson2
import os
import numpy as np
import platform
import logging
from IPython import embed
import time

class Simulator:
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
                 auto_sync=True,
                 optimized_renderer=False,
                 env_texture_filename=None,
                 skybox_size=20.,
                 rendering_settings=MeshRendererSettings()
                 vrFullscreen=True,
		         vrEyeTracking=False,
                 vrMode=True):

        """
        Simulator class is a wrapper of physics simulator (pybullet) and MeshRenderer, it loads objects into
        both pybullet and also MeshRenderer and syncs the pose of objects and robot parts.

        :param gravity: gravity on z direction.
        :param physics_timestep: timestep of physical simulation, p.stepSimulation()
        :param render_timestep: timestep of rendering, and Simulator.step() function
        :param mode: choose mode from gui, headless, iggui (only open iGibson UI), or pbgui(only open pybullet UI)
        :param image_width: width of the camera image
        :param image_height: height of the camera image
        :param vertical_fov: vertical field of view of the camera image in degrees
        :param device_idx: GPU device index to run rendering on
        :param render_to_tensor: Render to GPU tensors
        :param auto_sync: automatically sync object poses to gibson renderer, by default true,
        disable it when you want to run multiple physics step but don't need to visualize each frame
        """
        print("Starting init!")
        # physics simulator
        self.gravity = gravity
        self.physics_timestep = physics_timestep
        self.render_timestep = render_timestep
        self.mode = mode

        #Todo: eliminate this
        self.objects = []

        plt = platform.system()
        if plt == 'Darwin' and self.mode == 'gui':
            self.mode = 'iggui'  # for mac os disable pybullet rendering
            logging.warn('Rendering both iggui and pbgui is not supported on mac, choose either pbgui or '
                         'iggui. Default to iggui.')

        self.use_pb_renderer = False
        self.use_ig_renderer = False
        self.use_vr_renderer = False
        
        if self.mode in ['gui', 'iggui']:
            self.use_ig_renderer = True

        if self.mode in ['gui', 'pbgui']:
            self.use_pb_renderer = True

        if self.mode in ['vr']:
            self.use_vr_renderer = True
                   
        # renderer
        self.vrFullscreen = vrFullscreen
        self.vrEyeTracking = vrEyeTracking
        self.vrMode = vrMode
        self.max_haptic_duration = 4000
        self.image_width = image_width
        self.image_height = image_height
        self.vertical_fov = vertical_fov
        self.device_idx = device_idx
        self.render_to_tensor = render_to_tensor
        self.auto_sync = auto_sync
        self.optimized_renderer = rendering_settings.optimized
        self.rendering_settings = rendering_settings
        self.load()

    def set_timestep(self, physics_timestep, render_timestep):
        """
        :param timestep: set timestep after the initialization of Simulator
        """
        self.physics_timestep = physics_timestep
        self.render_timestep = render_timestep
        p.setTimeStep(self.physics_timestep)

    def add_viewer(self):
        """
        Attach a debugging viewer to the renderer. This will make the step much slower so should be avoided when
        training agents
        """
        if self.use_vr_renderer:
            self.viewer = ViewerVR()
        else:
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
                                            rendering_settings=self.rendering_settings)
        elif self.use_vr_renderer:
            if self.env_texture_filename is not None:
                self.renderer = MeshRendererVR(fullscreen=self.vrFullscreen,
                                            useEyeTracking=self.vrEyeTracking,
                                            vrMode=self.vrMode,
                                            env_texture_filename=self.env_texture_filename)
            else:
                self.renderer = MeshRendererVR(fullscreen=self.vrFullscreen,
                                            useEyeTracking=self.vrEyeTracking,
                                            vrMode=self.vrMode)
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

        if (self.use_ig_renderer or self.use_vr_renderer) and not self.render_to_tensor:
            self.add_viewer()

        self.visual_objects = {}
        self.robots = []
        self.scene = None
        self.next_class_id = 0

        if self.use_ig_renderer and not self.render_to_tensor:
            self.add_viewer()

    def optimize_vertex_and_texture(self):
        self.renderer.optimize_vertex_and_texture()

    def load_without_pybullet_vis(load_func):
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
                     class_id=None,
                     ):
        """
        Import a scene into the simulator. A scene could be a synthetic one or a realistic Gibson Environment.

        :param scene: Scene object
        :param texture_scale: Option to scale down the texture for rendering
        :param load_texture: If you don't need rgb output, texture loading could be skipped to make rendering faster
        :param class_id: Class id for rendering semantic segmentation
        """

        # Load the scene. Returns a list of pybullet ids of the objects loaded that we can use to
        # load them in the renderer
        new_object_pb_ids = scene.load()
        self.objects += new_object_pb_ids

        # Load the objects in the renderer
        for new_object_pb_id in new_object_pb_ids:
            self.load_object_in_renderer(new_object_pb_id, class_id=class_id, texture_scale=texture_scale,
                                         load_texture=load_texture, render_floor_plane=render_floor_plane)

        self.scene = scene
        return new_object_pb_ids

    @load_without_pybullet_vis
    def import_ig_scene(self, scene):
        """
        Import scene from iGSDF class
        :param scene: iGSDFScene instance
        :return: ids from scene.load function
        """
        new_object_ids = scene.load()
        self.objects += new_object_ids
        if scene.texture_randomization:
            # use randomized texture


            for body_id, visual_mesh_to_material in \
                    zip(new_object_ids, scene.visual_mesh_to_material):
                shadow_caster = True
                if scene.objects_by_id[body_id].category == 'ceilings':
                    shadow_caster = False

                self.load_articulated_object_in_renderer(
                    body_id, class_id=body_id,
                    visual_mesh_to_material=visual_mesh_to_material,
                    shadow_caster=shadow_caster)
        else:
            # use default texture
            for body_id in new_object_ids:
                use_pbr = True
                use_pbr_mapping = True
                shadow_caster = True

                if scene.objects_by_id[body_id].category in ['walls', 'floors', 'ceilings']:
                    use_pbr = False
                    use_pbr_mapping = False
                if scene.objects_by_id[body_id].category == 'ceilings':
                    shadow_caster = False
                self.load_articulated_object_in_renderer(body_id, class_id=body_id,
                                                         use_pbr=use_pbr,
                                                         use_pbr_mapping=use_pbr_mapping,
                                                         shadow_caster=shadow_caster)
        self.scene = scene

        return new_object_ids

    @load_without_pybullet_vis
    def import_object(self, obj, class_id=None, use_pbr=True, use_pbr_mapping=True, shadow_caster=True):
        """
        Import a non-articulated object into the simulator

        :param obj: Object to load
        :param class_id: Class id for rendering semantic segmentation
        """

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

        if class_id is None:
            class_id = self.next_class_id
        self.next_class_id += 1

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

        if class_id is None:
            class_id = self.next_class_id
        self.next_class_id += 1

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
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
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
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
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
                                         poses_rot=poses_rot,
                                         poses_trans=poses_trans,
                                         dynamic=True,
                                         robot=None,
                                         use_pbr=use_pbr,
                                         use_pbr_mapping=use_pbr_mapping,
                                         shadow_caster=shadow_caster)

    @load_without_pybullet_vis
    def import_robot(self, robot, class_id=None):
        """
        Import a robot into the simulator

        :param robot: Robot
        :param class_id: Class id for rendering semantic segmentation
        :return: pybullet id
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
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
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
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
                link_ids.append(link_id)
            elif type == p.GEOM_BOX:
                filename = os.path.join(
                    gibson2.assets_path, 'models/mjcf_primitives/cube.obj')
                self.renderer.load_object(filename,
                                          transform_orn=rel_orn,
                                          transform_pos=rel_pos,
                                          input_kd=color[:3],
                                          scale=np.array(dimensions))
                visual_objects.append(len(self.renderer.get_visual_objects()) - 1)
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

        if self.auto_sync:
            self.sync()

    def sync(self):
        """
        Update positions in renderer without stepping the simulation. Usually used in the reset() function
        """
        for instance in self.renderer.get_instances():
            if instance.dynamic:
                self.update_position(instance)
        if (self.use_ig_renderer or self.use_vr_renderer) and not self.viewer is None:
            self.viewer.update()
    
    # Returns event data as list of lists. Each sub-list contains deviceType and eventType. List is empty is all 
    # events are invalid. 
    # deviceType: left_controller, right_controller
    # eventType: grip_press, grip_unpress, trigger_press, trigger_unpress, touchpad_press, touchpad_unpress,
    # touchpad_touch, touchpad_untouch, menu_press, menu_unpress (menu is the application button)
    def pollVREvents(self):
        if not self.use_vr_renderer:
            return []

        eventData = self.renderer.vrsys.pollVREvents()
        return eventData

    # Call this after step - returns all VR device data for a specific device
    # Device can be hmd, left_controller or right_controller
    # Returns isValid (indicating validity of data), translation and rotation in Gibson world space
    def getDataForVRDevice(self, deviceName):
        if not self.use_vr_renderer:
            return [None, None, None]

        # Use fourth variable in list to get actual hmd position in space
        isValid, translation, rotation, _ = self.renderer.vrsys.getDataForVRDevice(deviceName)
        return [isValid, translation, rotation]

    # Get world position of HMD without offset
    def getHmdWorldPos(self):
        if not self.use_vr_renderer:
            return None
        
        _, _, _, hmd_world_pos = self.renderer.vrsys.getDataForVRDevice('hmd')
        return hmd_world_pos

    # Call this after getDataForVRDevice - returns analog data for a specific controller
    # Controller can be left_controller or right_controller
    # Returns trigger_fraction, touchpad finger position x, touchpad finger position y
    # Data is only valid if isValid is true from previous call to getDataForVRDevice
    # Trigger data: 1 (closed) <------> 0 (open)
    # Analog data: X: -1 (left) <-----> 1 (right) and Y: -1 (bottom) <------> 1 (top)
    def getButtonDataForController(self, controllerName):
        if not self.use_vr_renderer:
            return [None, None, None]
        
        trigger_fraction, touch_x, touch_y = self.renderer.vrsys.getButtonDataForController(controllerName)
        return [trigger_fraction, touch_x, touch_y]
    
    # Returns eye tracking data as list of lists. Order: is_valid, gaze origin, gaze direction, gaze point, left pupil diameter, right pupil diameter (both in millimeters)
    # Call after getDataForVRDevice, to guarantee that latest HMD transform has been acquired
    def getEyeTrackingData(self):
        if not self.use_vr_renderer or not self.vrEyeTracking:
            return [None, None, None, None, None]
            
        is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter = self.renderer.vrsys.getEyeTrackingData()
        return [is_valid, origin, dir, left_pupil_diameter, right_pupil_diameter]

    # Sets the translational offset of the VR system (HMD, left controller, right controller)
    # Can be used for many things, including adjusting height and teleportation-based movement
    # Input must be a list of three floats, corresponding to x, y, z in Gibson coordinate space
    def setVROffset(self, pos=None):
        if not self.use_vr_renderer:
            return

        self.renderer.vrsys.setVROffset(-pos[1], pos[2], -pos[0])

    # Gets the current VR offset vector in list form: x, y, z (in Gibson coordinates)
    def getVROffset(self):
        if not self.use_vr_renderer:
            return [None, None, None]

        x, y, z = self.renderer.vrsys.getVROffset()
        return [x, y, z]

    # Gets the direction vectors representing the device's coordinate system in list form: x, y, z (in Gibson coordinates)
    # List contains "right", "up" and "forward" vectors in that order
    # Device can be one of "hmd", "left_controller" or "right_controller"
    def getDeviceCoordinateSystem(self, device):
        if not self.use_vr_renderer:
            return [None, None, None]

        vec_list = []

        coordinate_sys = self.renderer.vrsys.getDeviceCoordinateSystem(device)
        for dir_vec in coordinate_sys:
            vec_list.append(dir_vec)

        return vec_list

    # Triggers a haptic pulse of the specified strength (0 is weakest, 1 is strongest)
    # Device can be one of "hmd", "left_controller" or "right_controller"
    def triggerHapticPulse(self, device, strength):
        if not self.use_vr_renderer:
            print("Error: can't use haptics without VR system!")
        else:
            self.renderer.vrsys.triggerHapticPulseForDevice(device, int(self.max_haptic_duration * strength))

    @staticmethod
    def update_position(instance):
        """
        Update position for an object or a robot in renderer.

        :param instance: Instance in the renderer
        """
        if isinstance(instance, Instance):
            # pos and orn of the inertial frame of the base link,
            # instead of the base link frame
            pos, orn = p.getBasePositionAndOrientation(
                instance.pybullet_uuid)

            # Need to convert to the base link frame because that is
            # what our own renderer keeps track of
            # Based on pyullet docuementation:
            # urdfLinkFrame = comLinkFrame * localInertialFrame.inverse().
            _, _, _, inertial_pos, inertial_orn, _, _, _, _, _, _, _ = \
                p.getDynamicsInfo(instance.pybullet_uuid, -1)
            inv_inertial_pos, inv_inertial_orn =\
                p.invertTransform(inertial_pos, inertial_orn)
            # Now pos and orn are converted to the base link frame
            pos, orn = p.multiplyTransforms(
                pos, orn, inv_inertial_pos, inv_inertial_orn)
            instance.set_position(pos)
            instance.set_rotation(xyzw2wxyz(orn))
        elif isinstance(instance, InstanceGroup):
            poses_rot = []
            poses_trans = []
            for link_id in instance.link_ids:
                if link_id == -1:
                    # same conversion is needed as above
                    pos, orn = p.getBasePositionAndOrientation(
                        instance.pybullet_uuid)
                    _, _, _, inertial_pos, inertial_orn, _, _, _, _, _, _, _ = \
                        p.getDynamicsInfo(instance.pybullet_uuid, -1)
                    inv_inertial_pos, inv_inertial_orn =\
                        p.invertTransform(inertial_pos, inertial_orn)
                    pos, orn = p.multiplyTransforms(
                        pos, orn, inv_inertial_pos, inv_inertial_orn)
                else:
                    _, _, _, _, pos, orn = p.getLinkState(
                        instance.pybullet_uuid, link_id)
                poses_rot.append(np.ascontiguousarray(
                    quat2rotmat(xyzw2wxyz(orn))))
                poses_trans.append(np.ascontiguousarray(xyz2mat(pos)))

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
            print("******************PyBullet Logging Information:")
            p.resetSimulation(physicsClientId=self.cid)
            p.disconnect(self.cid)
            print("PyBullet Logging Information******************")
        self.renderer.release()
