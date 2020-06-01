import numpy as np
from gibson2.core.render.mesh_renderer.Release import MeshRendererContext
import matplotlib.pyplot as plt

# VR wrapper class on top of Gibson Mesh Renderers
class MeshRendererVR():
    # Init takes in a renderer type to use for VR (which can be of type MeshRenderer or something else as long as it conforms to the same interface)
    def __init__(self, rendererType, vrWidth=None, vrHeight=None, msaa=False, optimize=True, vrMode=True):
        self.msaa = msaa
        self.optimize = optimize
        self.vrMode = vrMode
        self.vrsys = MeshRendererContext.VRSystem()
        # Default recommended is 2016 x 2240
        # self.width, self.height = self.vrsys.initVR()a
        self.baseWidth = 1080
        self.baseHeight = 1200
        self.scaleFactor = 1.4
        self.width = int(self.baseWidth * self.scaleFactor)
        self.height = int(self.baseHeight * self.scaleFactor)
        if vrWidth is not None and vrHeight is not None:
            self.renderer = rendererType(width=vrWidth, height=vrHeight, msaa=self.msaa, shouldHideWindow=False, optimize=self.optimize)
        else:
            self.renderer = rendererType(width=self.width, height=self.height, msaa=self.msaa, shouldHideWindow=False, optimize=self.optimize)

        self.fig = plt.figure()

    # Sets the position of the VR camera to the position argument given
    def set_vr_camera(self, pos):
        # Gibson coordinate system is rotated from OpenGL
        # So we map (vr from gib) x<-y, y<-z and z<-x
        self.vrsys.setVRCamera(-pos[1], pos[2], -pos[0])

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
                     dynamic=False,
                     softbody=False):
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

    # Optimizes vertex and texture
    def optimize_vertex_and_texture(self):
        self.renderer.optimize_vertex_and_texture()

    # Renders VR scenes and returns the left eye frame
    def render(self):
        if not self.vrMode:
            self.renderer.render(modes=('rgb'), shouldReadBuffer=False)
        else:
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

        if self.vrMode:
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

    # Return instances stored in renderer
    def get_instances(self):
        return self.renderer.get_instances()
    
    # Return visual objects stored in renderer
    def get_visual_objects(self):
        return self.renderer.get_visual_objects()

    # Releases VR system and renderer
    def release(self):
        self.renderer.release()
        self.vrsys.releaseVR()