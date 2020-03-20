import numpy as np
from VRUtils import VRSystem

class MeshRendererVR():
    # Init takes in a renderer type to use for VR (which can be of type MeshRenderer or something else as long as it conforms to the same interface)
    def __init__(self, rendererType):
        self.vrsys = VRSystem()
        recommendedWidth, recommendedHeight = self.vrsys.initVR()

        # Need GLFW context since VR mesh renderer runs on Windows
        self.renderer = rendererType(width=recommendedWidth, height=recommendedHeight)
    
        # Variables used for debugging the HMD display
        self.colorFbo = None
        self.colorTexId = None

        print("Finished init!")

    # Sets the position of the VR camera to the position argument given
    def set_vr_camera(self, pos):
        self.vrsys.setVRCamera(pos[0], pos[1], pos[2])

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

    # Debugging function for creating simple color framebuffer in PyOpenGL
    def setupDebugFramebuffer(self):
        self.colorFbo = GL.glGenFramebuffers(1)
        self.colorTexId = GL.glGenTextures(1)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.colorTexId)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, self.renderer.width, self.renderer.height, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.colorFbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.colorTexId, 0)
        GL.glViewport(0, 0, self.renderer.width, self.renderer.height)
        GL.glDrawBuffers(1, [GL.GL_COLOR_ATTACHMENT0])

        assert GL.glCheckFramebufferStatus(GL.GL_FRAMEBUFFER) == GL.GL_FRAMEBUFFER_COMPLETE

    # Debugging function for rendering simple color framebuffer in PyOpenGL
    def renderDebugFramebuffer(self):
        GL.glClearColor(1.0, 0.0, 0.0, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)

        self.vrsys.postRenderVRForEye("left", self.colorTexId)
        self.vrsys.postRenderVRForEye("right", self.colorTexId)
        self.vrsys.postRenderVRUpdate(True)

    # Renders VR scenes and returns the left eye frame
    # vrMode boolean causes the renderer to operate like a non-VR renderer for debugging purposes
    def render(self, vrMode=True):
        if not vrMode:
            return self.renderer.render(modes=('rgb'))

        leftProj, leftView, rightProj, rightView = self.vrsys.preRenderVR()

        # Render and submit left eye
        self.renderer.V = leftView
        self.renderer.P = leftProj
        leftFrames = self.renderer.render(modes=('rgb'))
        #self.vrsys.postRenderVRForEye("left", self.renderer.color_tex_rgb)

        # Render and submit right eye
        self.renderer.V = rightView
        self.renderer.P = rightProj
        self.renderer.render(modes=('rgb'))
        #self.vrsys.postRenderVRForEye("right", self.renderer.color_tex_rgb)

        # Currently hands off to the compositor
        self.vrsys.postRenderVRUpdate(True)

        return leftFrames

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
        renderer.release()
        self.vrsys.releaseVR()