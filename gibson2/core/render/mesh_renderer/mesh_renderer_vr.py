import numpy as np
from gibson2.core.render.mesh_renderer.Release import MeshRendererContext
import matplotlib.pyplot as plt

# VR wrapper class on top of Gibson Mesh Renderers
class MeshRendererVR():
    # Init takes in a renderer type to use for VR (which can be of type MeshRenderer or something else as long as it conforms to the same interface)
    def __init__(self, rendererType, vrWidth=None, vrHeight=None, msaa=False, fullscreen=True, optimize=True, useEyeTracking=False, vrMode=True):
        # Msaa slows down VR significantly, so only use it if you have to
        self.msaa = msaa
        self.fullscreen = fullscreen
        self.optimize = optimize
        self.useEyeTracking = useEyeTracking
        self.vrMode = vrMode
        self.vrsys = MeshRendererContext.VRSystem()
        # Default recommended is 2016 x 2240
        if self.vrMode:
            self.recWidth, self.recHeight = self.vrsys.initVR(self.useEyeTracking)
        self.baseWidth = 1080
        self.baseHeight = 1200
        self.scaleFactor = 1.4
        self.width = int(self.baseWidth * self.scaleFactor)
        self.height = int(self.baseHeight * self.scaleFactor)
        if vrWidth is not None and vrHeight is not None:
            self.renderer = rendererType(width=vrWidth, height=vrHeight, msaa=self.msaa, useGlfwWindow=True, fullscreen=self.fullscreen, optimize=self.optimize)
        else:
            self.renderer = rendererType(width=self.width, height=self.height, msaa=self.msaa, useGlfwWindow=True, fullscreen=self.fullscreen, optimize=self.optimize)

        self.fig = plt.figure()

    # Sets the position of the VR system (HMD, left controller, right controller).
    # Can be used for many things, including adjusting height and teleportation-based movement
    def set_vr_position(self, pos):
        # Gibson coordinate system is rotated from OpenGL
        # So we map (vr from gib) x<-y, y<-z and z<-x
        self.vrsys.setVRPosition(-pos[1], pos[2], -pos[0])

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
        if self.vrMode:
            leftProj, leftView, rightProj, rightView = self.vrsys.preRenderVR()

            # Render and submit left eye
            self.renderer.V = leftView
            self.renderer.P = leftProj
            
            self.renderer.render(modes=('rgb'))
            self.vrsys.postRenderVRForEye("left", self.renderer.color_tex_rgb)
            # Render and submit right eye
            self.renderer.V = rightView
            self.renderer.P = rightProj
            
            self.renderer.render(modes=('rgb'))
            self.vrsys.postRenderVRForEye("right", self.renderer.color_tex_rgb)

            # TODO: Experiment with this boolean for handoff
            self.vrsys.postRenderVRUpdate(False)
        else:
            self.renderer.render(modes=('rgb'))

    # Render companion window - renders right eye in VR
    def render_companion_window(self):
        self.renderer.render_companion_window()
        
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