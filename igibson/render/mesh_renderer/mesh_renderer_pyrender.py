import logging
import platform

import numpy as np
import pyrender
import trimesh
from PIL import Image

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.mesh_renderer.text import TextManager
from igibson.robots.robot_base import BaseRobot
from igibson.utils.constants import AVAILABLE_MODALITIES, MAX_INSTANCE_COUNT
from igibson.utils.mesh_util import lookat, mat2xyz, ortho, quat2rotmat, safemat2quat, xyz2mat, xyzw2wxyz

log = logging.getLogger(__name__)

Image.MAX_IMAGE_PIXELS = None
NO_MATERIAL_DEFINED_IN_SHAPE_AND_NO_OVERWRITE_SUPPLIED = -1


class MeshRendererPyRender(object):
    """
    MeshRenderer is a lightweight OpenGL renderer.
    It manages a set of visual objects, and instances of those objects.
    It also manage a device to create OpenGL context on, and create buffers to store rendering results.
    """

    def __init__(
        self,
        width=512,
        height=512,
        vertical_fov=90,
        device_idx=0,
        rendering_settings=MeshRendererSettings(),
        simulator=None,
    ):
        """
        :param width: width of the renderer output
        :param height: width of the renderer output
        :param vertical_fov: vertical field of view for the renderer
        :param device_idx: which GPU to run the renderer on
        :param rendering_settings: rendering settings
        :param simulator: simulator object
        """
        self.simulator = simulator
        self.rendering_settings = rendering_settings
        self.shaderProgram = None
        self.windowShaderProgram = None
        self.fbo = None
        self.color_tex_rgb, self.color_tex_normal, self.color_tex_semantics, self.color_tex_3d = None, None, None, None
        self.color_tex_scene_flow, self.color_tex_optical_flow, self.color_tex_ins_seg = None, None, None
        self.depth_tex = None
        self.VAOs = []
        self.VBOs = []
        self.textures = []
        self.objects = []
        self.visual_objects = []
        self.vertex_data = []
        self.shapes = []
        self.width = width
        self.height = height
        self.faces = []
        self.instances = []
        self.update_instance_id_to_pb_id_map()
        self.fisheye = rendering_settings.use_fisheye
        self.optimized = rendering_settings.optimized
        self.texture_files = {}
        self.enable_shadow = rendering_settings.enable_shadow
        self.platform = platform.system()
        self.optimization_process_executed = False
        self.pose_trans_array = None
        self.pose_rot_array = None
        self.last_trans_array = None
        self.last_rot_array = None
        self.lightP = ortho(-5, 5, -5, 5, -10, 20.0)
        # Manages text data that is shared across multiple Text instances
        self.texts = []

        self.msaa = rendering_settings.msaa

        self.scene = pyrender.Scene()

        self.colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self.lightcolor = [1, 1, 1]

        # default light looking down and tilted
        light_pose = trimesh.transformations.compose_matrix(translate=[0, 0, 2], angles=[np.pi, 0, 0])
        self.light = self.scene.add(pyrender.PointLight(self.lightcolor, intensity=3.0), pose=light_pose)

        self.vertical_fov = vertical_fov
        self.horizontal_fov = (
            2 * np.arctan(np.tan(self.vertical_fov / 180.0 * np.pi / 2.0) * self.width / self.height) / np.pi * 180.0
        )

        self.camera = [1, 0, 0]
        self.target = [0, 0, 0]
        self.up = [0, 0, 1]
        self.znear = 0.1
        self.zfar = 100
        V = lookat(self.camera, self.target, up=self.up)
        V = np.ascontiguousarray(V, np.float32)

        self.cam = self.scene.add(
            pyrender.PerspectiveCamera(
                yfov=self.vertical_fov, znear=self.znear, zfar=self.zfar, aspectRatio=self.width / self.height
            ),
            pose=V,
        )

        self.rr = pyrender.OffscreenRenderer(self.width, self.height)

        self.material_idx_to_material_instance_mapping = {}
        self.shape_material_idx = []
        # shape_material_idx is a list with the same length as self.shapes and self.VAOs, indicating the material_idx
        # that each shape is mapped to.
        # Number of unique shapes comprising the optimized renderer buffer
        self.or_buffer_shape_num = 0
        # Store trans and rot data for OR as a single variable that we update every frame - avoids copying variable each frame
        self.trans_data = None
        self.rot_data = None

        self.skybox_size = rendering_settings.skybox_size

    def set_light_position_direction(self, position, target):
        """
        Set light position and orientation.

        :param position: light position
        :param target: light target
        """
        V = lookat(position, target)
        V = np.ascontiguousarray(V, np.float32)
        self.scene.set_pose(self.light, V)

    def load_procedural_material(self, material, texture_scale):
        raise NotImplementedError("Not implemented")

    def load_randomized_material(self, material, texture_scale):
        raise NotImplementedError("Not implemented")

    def load_object(
        self,
        obj_path,
        scale=np.array([1, 1, 1]),
        transform_orn=None,
        transform_pos=None,
        input_kd=None,
        texture_scale=1.0,
        overwrite_material=None,
    ):
        """
        Load a wavefront obj file into the renderer and create a VisualObject to manage it.

        :param obj_path: path of obj file
        :param scale: scale, default 1
        :param transform_orn: rotation quaternion, convention xyzw
        :param transform_pos: translation for loading, it is a list of length 3
        :param input_kd: if loading material fails, use this default material. input_kd should be a list of length 3
        :param texture_scale: texture scale for the object, downsample to save memory
        :param overwrite_material: whether to overwrite the default Material (usually with a RandomizedMaterial for material randomization)
        :return: VAO_ids
        """
        mesh = pyrender.Mesh.from_trimesh(trimesh.load(obj_path))
        self.visual_objects.append(mesh)
        return len(self.visual_objects) - 1

    def add_instance_group(
        self,
        object_ids,
        link_ids=[-1],
        poses_trans=[np.eye(4)],
        poses_rot=[np.eye(4)],
        pybullet_uuid=None,
        ig_object=None,
        class_id=0,
        dynamic=False,
        softbody=False,
        use_pbr=True,
        use_pbr_mapping=True,
        shadow_caster=True,
        parent_body_name=None,
    ):
        """
        Create an instance group for a list of visual objects and link it to pybullet.

        :param object_ids: object ids of the visual objects
        :param link_ids: link_ids in pybullet
        :param poses_trans: initial translations for each visual object
        :param poses_rot: initial rotation matrix for each visual object
        :param pybullet_uuid: body id in pybullet
        :param ig_object: iGibson object associated with this instance group
        :param class_id: class_id to render semantics
        :param dynamic: whether the instance group is dynamic
        :param use_pbr: whether to use PBR
        :param use_pbr_mapping: whether to use PBR mapping
        :param shadow_caster: whether to cast shadow
        """

        if self.optimization_process_executed and self.optimized:
            log.error("Using optimized renderer and optimization process is already excuted, cannot add new " "objects")
            return

        use_pbr = use_pbr and self.rendering_settings.enable_pbr
        use_pbr_mapping = use_pbr_mapping and self.rendering_settings.enable_pbr

        instance_group = PyRenderInstanceGroup(
            self,
            [self.visual_objects[object_id] for object_id in object_ids],
            id=len(self.instances),
            link_ids=link_ids,
            pybullet_uuid=pybullet_uuid,
            ig_object=ig_object,
            class_id=class_id,
            poses_trans=poses_trans,
            poses_rot=poses_rot,
        )
        self.instances.append(instance_group)
        self.update_instance_id_to_pb_id_map()

    def set_camera(self, camera, target, up, cache=False):
        """
        Set camera pose.

        :param camera: camera position
        :param target: camera target
        :param up: up direction
        :param cache: whether to cache pose
        """
        self.camera = camera
        self.target = target
        self.up = up

        V = lookat(self.camera, self.target, up=self.up)
        V = np.ascontiguousarray(V, np.float32)
        self.scene.set_pose(self.cam, V)
        # change shadow mapping camera to be above the real camera
        # TODO reenable
        self.set_light_position_direction([self.camera[0], self.camera[1], 10], [self.camera[0], self.camera[1], 0])

    def render(
        self, modes=AVAILABLE_MODALITIES, hidden=(), return_buffer=True, render_shadow_pass=True, render_text_pass=True
    ):
        """
        A function to render all the instances in the renderer and read the output from framebuffer.

        :param modes: a tuple consisting of a subset of ('rgb', 'normal', 'seg', '3d', 'scene_flow', 'optical_flow')
        :param hidden: hidden instances to skip. When rendering from a robot's perspective, it's own body can be hidden
        :param return_buffer: whether to return the frame buffers as numpy arrays
        :param render_shadow_pass: whether to render shadow
        :return: a list of float32 numpy arrays of shape (H, W, 4) corresponding to `modes`, where last channel is alpha
        """
        # hide the objects that specified in hidden for optimized renderer
        # non-optimized renderer handles hidden objects in a different way
        if isinstance(modes, str):
            modes = (modes,)
        if modes != ("rgb",):
            raise ValueError("Not implemented")

        color = self.rr.render(self.scene)[0]
        return [np.concatenate([color, np.ones((self.height, self.width, 1))], axis=-1).astype(np.float32)]

    def get_instances(self):
        """
        Return instances.
        """
        return self.instances

    def transform_vector(self, vec):
        vec = np.array(vec)
        zeros = np.zeros_like(vec)

        vec_t = self.transform_point(vec)
        zero_t = self.transform_point(zeros)

        v = vec_t - zero_t
        return v

    def transform_point(self, vec):
        vec = np.array(vec)
        if vec.shape[0] == 3:
            v = self.scene.get_pose(self.cam).dot(np.concatenate([vec, np.array([1])]))
            return v[:3] / v[-1]
        elif vec.shape[0] == 4:
            v = self.scene.get_pose(self.cam).dot(vec)
            return v / v[-1]
        else:
            return None

    def transform_pose(self, pose):
        """
        Transform pose from world frame to camera frame.

        :param pose: pose in world frame
        :return: pose in camera frame
        """
        pose_rot = quat2rotmat(pose[3:])
        pose_trans = xyz2mat(pose[:3])
        pose_cam = self.scene.get_pose(self.cam).dot(pose_trans.T).dot(pose_rot).T
        return np.concatenate([mat2xyz(pose_cam), safemat2quat(pose_cam[:3, :3].T)])

    def render_active_cameras(self, modes=("rgb"), cache=True):
        """
        Render camera images for the active cameras. This is applicable for robosuite integration with iGibson,
        where there are multiple cameras defined but only some are active (e.g., to switch between views with TAB).

        :return: a list of frames (number of modalities x number of robots)
        """
        frames = []
        hide_robot = self.rendering_settings.hide_robot
        need_flow_info = "optical_flow" in modes or "scene_flow" in modes
        has_set_camera = False
        for instance in self.instances:
            if isinstance(instance.ig_object, BaseRobot):
                for camera in instance.ig_object.cameras:
                    if camera.is_active():
                        camera_pose = camera.get_pose()
                        camera_pos = camera_pose[:3]
                        camera_ori = camera_pose[3:]
                        camera_ori_mat = quat2rotmat([camera_ori[-1], camera_ori[0], camera_ori[1], camera_ori[2]])[
                            :3, :3
                        ]
                        camera_view_dir = camera_ori_mat.dot(np.array([0, 0, -1]))  # Mujoco camera points in -z
                        if need_flow_info and has_set_camera:
                            raise ValueError("We only allow one robot in the scene when rendering optical/scene flow.")
                        self.set_camera(
                            camera_pos, camera_pos + camera_view_dir, [0, 0, 1], cache=need_flow_info and cache
                        )
                        has_set_camera = True
                        for item in self.render(modes=modes, hidden=[[], [instance]][hide_robot]):
                            frames.append(item)
        return frames

    def render_robot_cameras(self, modes=("rgb"), cache=True):
        """
        Render robot camera images.

        :param modes: a tuple of modalities to render
        :param cache: if cache is True, cache the robot pose for optical flow and scene flow calculation.
        One simulation step can only have one rendering call with cache=True

        :return: a list of frames (number of modalities x number of robots)
        """
        frames = []
        need_flow_info = "optical_flow" in modes or "scene_flow" in modes
        if need_flow_info and len(self.simulator.scene.robots) > 1:
            raise ValueError("We only allow one robot in the scene when rendering optical/scene flow.")

        for robot in self.simulator.scene.robots:
            frames.extend(self.render_single_robot_camera(robot, modes=modes, cache=cache))

        return frames

    def render_single_robot_camera(self, robot, modes=("rgb"), cache=True):
        frames = []
        hide_instances = robot.renderer_instances if self.rendering_settings.hide_robot else []
        need_flow_info = "optical_flow" in modes or "scene_flow" in modes
        camera_pos = robot.eyes.get_position()
        orn = robot.eyes.get_orientation()
        mat = quat2rotmat(xyzw2wxyz(orn))[:3, :3]
        view_direction = mat.dot(np.array([1, 0, 0]))
        up_direction = mat.dot(np.array([0, 0, 1]))
        self.set_camera(camera_pos, camera_pos + view_direction, up_direction, cache=need_flow_info and cache)
        for item in self.render(modes=modes, hidden=hide_instances):
            frames.append(item)

        return frames

    def update_instance_id_to_pb_id_map(self):
        self.instance_id_to_pb_id = np.full((MAX_INSTANCE_COUNT,), -1)
        for inst in self.instances:
            self.instance_id_to_pb_id[inst.id] = inst.pybullet_uuid if inst.pybullet_uuid is not None else -1

    def get_pb_ids_for_instance_ids(self, instance_ids):
        return self.instance_id_to_pb_id[instance_ids.astype(int)]


class PyRenderInstanceGroup(object):
    """
    InstanceGroup is a set of visual objects.
    It is grouped together because they are kinematically connected.
    Robots and articulated objects are represented as instance groups.
    """

    def __init__(
        self,
        renderer,
        objects,
        id,
        link_ids,
        pybullet_uuid,
        ig_object,
        class_id,
        poses_trans,
        poses_rot,
    ):
        """
        :param objects: visual objects
        :param id: id of this instance_group
        :param link_ids: link_ids in pybullet
        :param pybullet_uuid: body id in pybullet
        :param class_id: class_id to render semantics
        :param poses_trans: initial translations for each visual object
        :param poses_rot: initial rotation matrix for each visual object
        :param dynamic: whether the instance group is dynamic
        :param softbody: whether the instance group is for a soft body
        :param use_pbr: whether to use PBR
        :param use_pbr_mapping: whether to use PBR mapping
        :param shadow_caster: whether to cast shadow
        :param parent_body_name: name of the parent body, if any, to be used by the Robosuite/Mujoco bridge
        """
        self.objects = objects
        self.id = id
        self.link_ids = link_ids
        self.class_id = class_id
        self.renderer: MeshRendererPyRender = renderer

        self.pybullet_uuid = pybullet_uuid
        self.ig_object = ig_object

        self.dynamic = True
        self.hidden = False
        self.highlight = False

        # TODO: Fix this terrible API
        self.nodes = []
        for mesh, pos, rot in zip(self.objects, poses_trans, poses_rot):
            pose = np.eye(4)
            pose[:3, :3] = rot[:3, :3]
            pose[:3, 3] = pos[:3, 3]
            self.nodes.append(self.renderer.scene.add(mesh, pose=pose))

    def set_highlight(self, highlight):
        pass

    def set_position_for_part(self, pos, j):
        """
        Set positions for one part of this InstanceGroup

        :param pos: position
        :param j: part index
        """
        pose = self.renderer.scene.get_pose(self.nodes[j])
        pose[:3, 3] = pos[:3, 3]
        self.renderer.scene.set_pose(self.nodes[j], pose)

    def set_rotation_for_part(self, rot, j):
        """
        Set rotations for one part of this InstanceGroup

        :param rot: rotation matrix
        :param j: part index
        """
        pose = self.renderer.scene.get_pose(self.nodes[j])
        pose[:3, :3] = rot[:3, :3]
        self.renderer.scene.set_pose(self.nodes[j], pose)

    def __str__(self):
        return "PyRenderInstanceGroup({}) -> Objects({})".format(
            self.id, ",".join([str(object.id) for object in self.objects])
        )

    def __repr__(self):
        return self.__str__()
