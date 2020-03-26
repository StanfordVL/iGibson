import numpy as np
import CGLUtils
from CGLUtils import VRSystem
from gibson2.core.render.mesh_renderer.glutils.meshutil import frustum

# VR wrapper class on top of Gibson Mesh Renderers
class MeshRendererVR():
    # Init takes in a renderer type to use for VR (which can be of type MeshRenderer or something else as long as it conforms to the same interface)
    def __init__(self, rendererType):
        self.vrsys = VRSystem()
        # Default recommended is 2016 x 2240
        self.width, self.height = self.vrsys.initVR()
        self.renderer = rendererType(width=self.width, height=self.height, shouldHideWindow=False)

        # Debugging variable for simple color test
        self.colorTex = None

        # Debugging image for testing VR compositor
        imgPath = "C:\\Users\\shen\\Desktop\\GibsonVRStuff\\vr_branch\\gibsonv2\\gibson2\\core\\render\\mesh_renderer\\good_boi.png"
        self.testTexId = CGLUtils.loadTextureWithAlpha(imgPath)
        print("Loaded good boi texture with id:")
        print(self.testTexId)

    # Sets the position of the VR camera to the position argument given
    def set_vr_camera(self, pos):
        # Gibson coordinate system is rotated from OpenGL
        # So we map (vr from gib) x<-y, y<-z and z<-x
        self.vrsys.setVRCamera(pos[1], pos[2], pos[0])

    # Resets the position of the VR camera
    def reset_vr_camera(self):
        self.vrsys.resetVRCamera()

    # Load object through renderer
    def load_object(self,
                    obj_path,
                    scale=np.array([1, 1, 1]),
                    transform_orn=None,
                    transform_pos=None,
                    input_kd=None,
                    texture_scale=1.0,
                    load_texture=True):
        self.renderer.load_object(obj_path, scale, transform_orn, transform_pos, input_kd, texture_scale, load_texture)

    # Add instance through renderer
    def add_instance(self,
                     object_id,
                     pybullet_uuid=None,
                     class_id=0,
                     pose_rot=np.eye(4),
                     pose_trans=np.eye(4),
                     dynamic=False):
        self.renderer.add_instance(object_id, pybullet_uuid, class_id, pose_rot, pose_trans, dynamic)

    # Add instance group through renderer
    def add_instance_group(self,
                           object_ids,
                           link_ids,
                           poses_rot,
                           poses_trans,
                           class_id=0,
                           pybullet_uuid=None,
                           dynamic=False,
                           robot=None):
        self.renderer.add_instance_group(object_ids, link_ids, poses_rot, poses_trans, class_id, pybullet_uuid, dynamic, robot)

    # Add robot through renderer
    def add_robot(self,
                  object_ids,
                  link_ids,
                  class_id,
                  poses_rot,
                  poses_trans,
                  pybullet_uuid=None,
                  dynamic=False,
                  robot=None):
        self.renderer.add_robot(object_ids, link_ids, class_id, poses_rot, poses_trans, pybullet_uuid, dynamic, robot)

    # Set up debugging framebuffer
    def setup_debug_framebuffer(self):
        self.colorFbo, self.colorTex = CGLUtils.setup_color_framebuffer(self.width, self.height)

    # Render debugging framebuffer
    def render_debug_framebuffer(self):
        leftProj, leftView, rightProj, rightView = self.vrsys.preRenderVR()

        CGLUtils.render_simple_color_to_fbo(self.colorFbo)

        self.vrsys.postRenderVRForEye("left", self.colorTex)
        self.vrsys.postRenderVRForEye("right", self.colorTex)

        self.renderer.r.flush_swap_glfw()

        self.vrsys.postRenderVRUpdate(False)
    
    # Test of loading an image with alpha
    def render_good_boi(self):
        leftProj, leftView, rightProj, rightView = self.vrsys.preRenderVR()

        self.vrsys.postRenderVRForEye("left", self.testTexId)
        self.vrsys.postRenderVRForEye("right", self.testTexId)

        # Boolean indicates whether system should hand off to compositor
        self.vrsys.postRenderVRUpdate(False)
    
    # Creates a frustum projection matrix from raw eye projection data from VR system
    def createProjMatrixFromRawValues(self, left, right, bottom, top, near, far):
        return frustum(left, right, bottom, top, near, far)

    # Renders VR scenes and returns the left eye frame
    def render(self):
        leftProj, leftView, rightProj, rightView = self.vrsys.preRenderVR()

        # Render and submit left eye
        self.renderer.V = leftView
        self.renderer.P = leftProj
        
        self.renderer.render(modes=('rgb'), shouldReadBuffer=False)
        self.vrsys.postRenderVRForEye("left", self.renderer.color_tex_rgb)

        # Render and submit right eye
        self.renderer.V = rightView
        self.renderer.P = rightProj
        
        self.renderer.render(modes=('rgb'), shouldReadBuffer=False)
        self.vrsys.postRenderVRForEye("right", self.renderer.color_tex_rgb)

        # Render companion window
        self.renderer.render_companion_window()

        # Boolean indicates whether system should hand off to compositor
        # TODO: Play around with this value
        self.vrsys.postRenderVRUpdate(False)

    # Sets camera position - only to be used in non-vr debugging mode
    def set_camera(self, camera, target, up):
        self.renderer.set_camera(camera, target, up)

    # Sets fov - only to be used in non-vr debugging mode
    def set_fov(self, fov):
        self.renderer.set_fov(fov)

    # Set position of light
    def set_light_pos(self, light):
        self.renderer.set_light_pos(light)

    # Get number of objects
    def get_num_objects(self):
        return self.renderer.get_num_objects()

    # Set pose of a specific object
    def set_pose(self, pose, idx):
        self.renderer.set_pose(pose, idx)

    # Releases VR system and renderer
    def release(self):
        self.renderer.release()
        self.vrsys.releaseVR()