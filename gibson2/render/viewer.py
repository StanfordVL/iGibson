import os
import random
import subprocess
from threading import Thread
import logging

import cv2
import numpy as np
import pybullet as p
from gibson2.objects.visual_marker import VisualMarker
from gibson2.render.mesh_renderer.mesh_renderer_vr import MeshRendererVR

class ViewerVR:
    def __init__(self):
        self.renderer = None
    
    def update(self):
        self.renderer.render()
        # Viewer is responsible for calling companion window rendering function
        self.renderer.render_companion_window()
from gibson2.utils.utils import rotate_vector_2d
import time


class Viewer:
    def __init__(self,
                 initial_pos=[0, 0, 1.2],
                 initial_view_direction=[1, 0, 0],
                 initial_up=[0, 0, 1],
                 simulator=None,
                 renderer=None,
                 min_cam_z=-1e6,
                 ):
        self.px = initial_pos[0]
        self.py = initial_pos[1]
        self.pz = initial_pos[2]
        self.theta = np.arctan2(
            initial_view_direction[1], initial_view_direction[0])
        self.phi = np.arctan2(initial_view_direction[2], np.sqrt(initial_view_direction[0] ** 2 +
                                                                 initial_view_direction[1] ** 2))
        self.min_cam_z = min_cam_z
        self.show_help = 0

        self._mouse_ix, self._mouse_iy = -1, -1
        self.left_down = False
        self.middle_down = False
        self.right_down = False
        self.view_direction = np.array(initial_view_direction)
        self.up = initial_up
        self.renderer = renderer
        self.simulator = simulator
        self.cid = []
        self.dist = 0

        # Flag to control if the mouse interface is in navigation or manipulation mode
        self.manipulation_mode = False

        # Video recording
        self.recording = False  # Boolean if we are recording frames from the viewer
        self.pause_recording = False  # Flag to pause/resume recording
        self.video_folder = ""

        cv2.namedWindow('ExternalView')
        cv2.moveWindow("ExternalView", 0, 0)
        cv2.namedWindow('RobotView')
        cv2.setMouseCallback('ExternalView', self.change_dir)
        self.create_visual_object()

    def create_visual_object(self):
        self.constraint_marker = VisualMarker(
            radius=0.04, rgba_color=[0, 0, 1, 1])
        self.constraint_marker2 = VisualMarker(visual_shape=p.GEOM_CAPSULE, radius=0.01, length=3,
                                               initial_offset=[0, 0, -1.5], rgba_color=[0, 0, 1, 1])
        print('SIMULATOR:', self.simulator)
        if self.simulator is not None:
            self.simulator.import_object(
                self.constraint_marker2, use_pbr=False)
            self.simulator.import_object(self.constraint_marker, use_pbr=False)
            self.constraint_marker.set_position([0, 0, -1])
            self.constraint_marker2.set_position([0, 0, -1])

    def apply_push_force(self, x, y, force):
        camera_pose = np.array([self.px, self.py, self.pz])
        self.renderer.set_camera(
            camera_pose, camera_pose + self.view_direction, self.up)
        #pos = self.renderer.get_3d_point(x,y)
        #frames = self.renderer.render(modes=('3d'))
        #position_cam = frames[0][y, x]
        # print(position_cam)
        position_cam = np.array([(x - self.renderer.width / 2) / float(self.renderer.width / 2) * np.tan(
            self.renderer.horizontal_fov / 2.0 / 180.0 * np.pi),
            -(y - self.renderer.height / 2) / float(self.renderer.height / 2) * np.tan(
            self.renderer.vertical_fov / 2.0 / 180.0 * np.pi),
            -1,
            1])
        position_cam[:3] *= 5

        position_world = np.linalg.inv(self.renderer.V).dot(position_cam)
        position_eye = camera_pose
        res = p.rayTest(position_eye, position_world[:3])
        # debug_line_id = p.addUserDebugLine(position_eye, position_world[:3], lineWidth=3)
        # and res[0][0] != self.marker.body_id:
        if len(res) > 0 and res[0][0] != -1:
            # there is hit
            object_id, link_id, _, hit_pos, hit_normal = res[0]
            p.changeDynamics(
                object_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
            p.applyExternalForce(
                object_id, link_id, -np.array(hit_normal) * force, hit_pos, p.WORLD_FRAME)

    def create_constraint(self, x, y, fixed=False):
        camera_pose = np.array([self.px, self.py, self.pz])
        self.renderer.set_camera(
            camera_pose, camera_pose + self.view_direction, self.up)
        # #pos = self.renderer.get_3d_point(x,y)
        # frames = self.renderer.render(modes=('3d'))
        # position_cam_org = frames[0][y, x]

        position_cam = np.array([(x - self.renderer.width / 2) / float(self.renderer.width / 2) * np.tan(
            self.renderer.horizontal_fov / 2.0 / 180.0 * np.pi),
            -(y - self.renderer.height / 2) / float(self.renderer.height / 2) * np.tan(
            self.renderer.vertical_fov / 2.0 / 180.0 * np.pi),
            -1,
            1])
        position_cam[:3] *= 5

        print(position_cam)
        position_world = np.linalg.inv(self.renderer.V).dot(position_cam)
        position_eye = camera_pose
        res = p.rayTest(position_eye, position_world[:3])
        if len(res) > 0 and res[0][0] != -1:
            object_id, link_id, _, hit_pos, hit_normal = res[0]
            p.changeDynamics(
                object_id, -1, activationState=p.ACTIVATION_STATE_WAKE_UP)
            link_pos, link_orn = None, None
            if link_id == -1:
                link_pos, link_orn = p.getBasePositionAndOrientation(object_id)
            else:
                link_state = p.getLinkState(object_id, link_id)
                link_pos, link_orn = link_state[:2]

            child_frame_pos, child_frame_orn = p.multiplyTransforms(*p.invertTransform(link_pos, link_orn), hit_pos,
                                                                    [0, 0, 0, 1])
            print(child_frame_pos)
            self.constraint_marker.set_position(hit_pos)
            self.constraint_marker2.set_position(hit_pos)
            self.dist = np.linalg.norm(np.array(hit_pos) - camera_pose)
            cid = p.createConstraint(
                parentBodyUniqueId=self.constraint_marker.body_id,
                parentLinkIndex=-1,
                childBodyUniqueId=object_id,
                childLinkIndex=link_id,
                jointType=[p.JOINT_POINT2POINT, p.JOINT_FIXED][fixed],
                jointAxis=(0, 0, 0),
                parentFramePosition=(0, 0, 0),
                childFramePosition=child_frame_pos,
                childFrameOrientation=child_frame_orn,
            )
            p.changeConstraint(cid, maxForce=100)
            self.cid.append(cid)
            self.interaction_x, self.interaction_y = x, y

    def remove_constraint(self):
        for cid in self.cid:
            p.removeConstraint(cid)
        self.cid = []
        self.constraint_marker.set_position([0, 0, 100])
        self.constraint_marker2.set_position([0, 0, 100])

    def move_constraint(self, x, y):
        # no constraint created but move_constraint called
        if len(self.cid) == 0:
            return
        camera_pose = np.array([self.px, self.py, self.pz])
        self.renderer.set_camera(
            camera_pose, camera_pose + self.view_direction, self.up)
        # #pos = self.renderer.get_3d_point(x,y)
        # frames = self.renderer.render(modes=('3d'))
        # position_cam_org = frames[0][y, x]

        position_cam = np.array([(x - self.renderer.width / 2) / float(self.renderer.width / 2) * np.tan(
            self.renderer.horizontal_fov / 2.0 / 180.0 * np.pi),
            -(y - self.renderer.height / 2) / float(self.renderer.height / 2) * np.tan(
            self.renderer.vertical_fov / 2.0 / 180.0 * np.pi),
            -1,
            1])
        position_cam[:3] = position_cam[:3] / \
            np.linalg.norm(position_cam[:3]) * self.dist
        position_world = np.linalg.inv(self.renderer.V).dot(position_cam)
        position_world /= position_world[3]
        self.constraint_marker.set_position(position_world[:3])
        self.constraint_marker2.set_position(position_world[:3])
        self.interaction_x, self.interaction_y = x, y

    def move_constraint_z(self, dy):
        # no constraint created but move_constraint called
        if len(self.cid) == 0:
            return
        x, y = self.interaction_x, self.interaction_y
        camera_pose = np.array([self.px, self.py, self.pz])
        self.renderer.set_camera(
            camera_pose, camera_pose + self.view_direction, self.up)
        # #pos = self.renderer.get_3d_point(x,y)
        # frames = self.renderer.render(modes=('3d'))
        # position_cam_org = frames[0][y, x]
        self.dist *= (1 - dy)
        if self.dist < 0.1:
            self.dist = 0.1
        position_cam = np.array([(x - self.renderer.width / 2) / float(self.renderer.width / 2) * np.tan(
            self.renderer.horizontal_fov / 2.0 / 180.0 * np.pi),
            -(y - self.renderer.height / 2) / float(self.renderer.height / 2) * np.tan(
            self.renderer.vertical_fov / 2.0 / 180.0 * np.pi),
            -1,
            1])
        position_cam[:3] = position_cam[:3] / \
            np.linalg.norm(position_cam[:3]) * self.dist
        position_world = np.linalg.inv(self.renderer.V).dot(position_cam)
        position_world /= position_world[3]
        self.constraint_marker.set_position(position_world[:3])
        self.constraint_marker2.set_position(position_world[:3])

    def change_dir(self, event, x, y, flags, param):
        if not self.manipulation_mode:
            if flags == cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_CTRLKEY and not self.right_down:
                # Only once, when pressing left mouse while cntrl key is pressed
                self._mouse_ix, self._mouse_iy = x, y
                self.right_down = True
            elif (event == cv2.EVENT_MBUTTONDOWN) or (flags == cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_SHIFTKEY and not self.middle_down):
                # Middle mouse button press or only once, when pressing left mouse while shift key is pressed (Mac
                # compatibility)
                self._mouse_ix, self._mouse_iy = x, y
                self.middle_down = True
            elif event == cv2.EVENT_LBUTTONDOWN:  # left mouse button press
                self._mouse_ix, self._mouse_iy = x, y
                self.left_down = True
            elif event == cv2.EVENT_LBUTTONUP:  # left mouse button released
                self.left_down = False
                self.right_down = False
                self.middle_down = False
            elif event == cv2.EVENT_MBUTTONUP:  # middle mouse button released
                self.middle_down = False

            if event == cv2.EVENT_MOUSEMOVE:  # moving mouse location on the window
                if self.left_down:  # if left button was pressed we change orientation of camera
                    dx = (x - self._mouse_ix) / 100.0
                    dy = (y - self._mouse_iy) / 100.0
                    self._mouse_ix = x
                    self._mouse_iy = y

                    if not ((flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_SHIFTKEY) or
                            (flags & cv2.EVENT_FLAG_CTRLKEY and flags & cv2.EVENT_FLAG_ALTKEY)):
                        self.phi += dy
                        self.phi = np.clip(
                            self.phi, -np.pi/2 + 1e-5, np.pi/2 - 1e-5)
                        self.theta += dx
                        self.view_direction = np.array([np.cos(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.cos(
                            self.phi), np.sin(self.phi)])

                elif self.middle_down:  # if middle button was pressed we get closer/further away in the viewing direction
                    d_vd = (y - self._mouse_iy) / 100.0
                    self._mouse_iy = y

                    motion_along_vd = d_vd*self.view_direction
                    self.px += motion_along_vd[0]
                    self.py += motion_along_vd[1]
                    self.pz += motion_along_vd[2]
                    self.pz = max(self.min_cam_z, self.pz)
                elif self.right_down:  # if right button was pressed we change translation of camera

                    zz = self.view_direction / \
                        np.linalg.norm(self.view_direction)
                    xx = np.cross(zz, np.array([0, 0, 1]))
                    xx = xx/np.linalg.norm(xx)
                    yy = np.cross(xx, zz)
                    motion_along_vx = -((x - self._mouse_ix) / 100.0)*xx
                    motion_along_vy = ((y - self._mouse_iy) / 100.0)*yy
                    self._mouse_ix = x
                    self._mouse_iy = y

                    self.px += (motion_along_vx[0] + motion_along_vy[0])
                    self.py += (motion_along_vx[1] + motion_along_vy[1])
                    self.pz += (motion_along_vx[2] + motion_along_vy[2])
                    self.pz = max(self.min_cam_z, self.pz)
        else:
            if (event == cv2.EVENT_MBUTTONDOWN) or (flags == cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_SHIFTKEY and not self.middle_down):
                # Middle mouse button press or only once, when pressing left mouse while shift key is pressed (Mac
                # compatibility)
                self._mouse_ix, self._mouse_iy = x, y
                self.middle_down = True
                self.create_constraint(x, y, fixed=True)
            elif event == cv2.EVENT_LBUTTONDOWN:  # left mouse button press
                self._mouse_ix, self._mouse_iy = x, y
                self.left_down = True
                self.create_constraint(x, y, fixed=False)
            elif event == cv2.EVENT_LBUTTONUP:  # left mouse button released
                self.left_down = False
                self.right_down = False
                self.middle_down = False
                self.remove_constraint()
            elif event == cv2.EVENT_MBUTTONUP:  # middle mouse button released
                self.left_down = False
                self.right_down = False
                self.middle_down = False
                self.remove_constraint()
            if event == cv2.EVENT_MOUSEMOVE:  # moving mouse location on the window
                if (self.left_down or self.middle_down) and not flags & cv2.EVENT_FLAG_CTRLKEY:
                    self._mouse_ix = x
                    self._mouse_iy = y
                    self.move_constraint(x, y)
                elif (self.left_down or self.middle_down) and flags & cv2.EVENT_FLAG_CTRLKEY:
                    dy = (y - self._mouse_iy) / 500.0
                    self.move_constraint_z(dy)

    def reset(self):
        self.px = 0
        self.py = 0
        self.pz = 1.2
        self.theta = 0
        self.phi = 0
        self.view_direction = np.array([np.cos(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.cos(
                            self.phi), np.sin(self.phi)])
        self.left_down = False
        self.right_down = False
        self.middle_down = False
        self.remove_constraint()


    def update(self):
        camera_pose = np.array([self.px, self.py, self.pz])
        if not self.renderer is None:
            self.renderer.set_camera(
                camera_pose, camera_pose + self.view_direction, self.up)

        if not self.renderer is None:
            frame = cv2.cvtColor(np.concatenate(self.renderer.render(modes=('rgb')), axis=1),
                                 cv2.COLOR_RGB2BGR)
        else:
            frame = np.zeros((300, 300, 3)).astype(np.uint8)

        # Text with the position and viewing direction of the camera of the external viewer
        text_color = (0, 0, 0)
        cv2.putText(frame, "px {:1.1f} py {:1.1f} pz {:1.1f}".format(self.px, self.py, self.pz), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, "[{:1.1f} {:1.1f} {:1.1f}]".format(*self.view_direction), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        cv2.putText(frame, ["nav mode", "manip mode"][self.manipulation_mode], (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        if self.show_help >= 0:
            if self.show_help >= 150:
                first_color = (255, 0, 0)
                help_text = "Keyboard cheatsheet:"
                cv2.putText(frame, help_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_color, 1, cv2.LINE_AA)
                help_text = "'w','a','s','d': up, left, down, right (any mode)"
                cv2.putText(frame, help_text, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_color, 1, cv2.LINE_AA)
                help_text = "'q','e': turn left, turn right (any mode)"
                cv2.putText(frame, help_text, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_color, 1, cv2.LINE_AA)
                help_text = "'m': Toggle mouse mode between navigation and manipulation"
                cv2.putText(frame, help_text, (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_color, 1, cv2.LINE_AA)
                help_text = "'r': Start/stop recording frames (results in \\tmp folder)"
                cv2.putText(frame, help_text, (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_color, 1, cv2.LINE_AA)
                help_text = "'p': Pause/resume recording"
                cv2.putText(frame, help_text, (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_color, 1, cv2.LINE_AA)
                help_text = "'h': Show this help on screen"
                cv2.putText(frame, help_text, (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_color, 1, cv2.LINE_AA)
                help_text = "'ESC': Quit"
                cv2.putText(frame, help_text, (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, first_color, 1, cv2.LINE_AA)
                self.show_help -= 1
            else:
                second_color = (255, 0, 255)
                help_text = "Mouse controls in navigation mode:"
                cv2.putText(frame, help_text, (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = "Left click and drag: rotate camera"
                cv2.putText(frame, help_text, (10, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = "CNTRL + left click and drag: translate camera left/right"
                cv2.putText(frame, help_text, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = "Middle click and drag (linux) or left SHIFT + left click and drag: up/down of mouse"
                cv2.putText(frame, help_text, (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = " translates camera forward/backwards"
                cv2.putText(frame, help_text, (10, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = "Mouse controls in manipulation mode:"
                cv2.putText(frame, help_text, (10, 180),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = "Left click and drag: create ball-joint connection to clicked object and move it"
                cv2.putText(frame, help_text, (10, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = "Middle click and drag (linux) or left SHIFT + left click and drag: create rigid connection"
                cv2.putText(frame, help_text, (10, 220),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = " to object and move it"
                cv2.putText(frame, help_text, (10, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                help_text = "CNTRL + click and drag: up/down of the mouse moves object further/closer"
                cv2.putText(frame, help_text, (10, 260),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, second_color, 1, cv2.LINE_AA)
                self.show_help -= 1
        cv2.imshow('ExternalView', frame)

        # We keep some double functinality for "backcompatibility"
        q = cv2.waitKey(1)
        move_vec = self.view_direction[:2]
        # step size is 0.05m
        move_vec = move_vec / np.linalg.norm(move_vec) * 0.05
        if q == ord('h'):
            self.show_help = 300
        elif q in [ord('w'), ord('s'), ord('a'), ord('d')]:
            if q == ord('w'):
                yaw = 0.0
            elif q == ord('s'):
                yaw = np.pi
            elif q == ord('a'):
                yaw = -np.pi / 2.0
            elif q == ord('d'):
                yaw = np.pi / 2.0
            move_vec = rotate_vector_2d(move_vec, yaw)
            self.px += move_vec[0]
            self.py += move_vec[1]
            if self.manipulation_mode:
                self.move_constraint(self._mouse_ix, self._mouse_iy)
        elif q == ord('q'):
            self.theta += np.pi/32
            self.view_direction = np.array([np.cos(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.cos(
                self.phi), np.sin(self.phi)])
            if self.manipulation_mode:
                self.move_constraint(self._mouse_ix, self._mouse_iy)
        elif q == ord('e'):
            self.theta -= np.pi/64
            self.view_direction = np.array([np.cos(self.theta) * np.cos(self.phi), np.sin(self.theta) * np.cos(
                self.phi), np.sin(self.phi)])
            if self.manipulation_mode:
                self.move_constraint(self._mouse_ix, self._mouse_iy)
        elif q == 27:
            if self.video_folder is not "":
                logging.info("You recorded a video. To compile the frames into a mp4 go to the corresponding subfolder" +
                             " in /tmp and execute: ")
                logging.info(
                    "ffmpeg -i %5d.png -y -c:a copy -c:v libx264 -crf 18 -preset veryslow -r 30 video.mp4")
                logging.info(
                    "The last folder you collected images for a video was: " + self.video_folder)
            exit()
        elif q == ord('r'):  # Start/Stop recording. Stopping saves frames to files
            if self.recording:
                self.recording = False
                self.pause_recording = False
            else:
                logging.info("Start recording*****************************")
                # Current time string to use to save the temporal urdfs
                timestr = time.strftime("%Y%m%d-%H%M%S")
                # Create the subfolder
                self.video_folder = os.path.join("/tmp",
                                                 '{}_{}_{}'.format(timestr, random.getrandbits(64), os.getpid()))
                os.makedirs(self.video_folder, exist_ok=True)
                self.recording = True
                self.frame_idx = 0
        elif q == ord('p'):  # Pause/Resume recording
            if self.pause_recording:
                self.pause_recording = False
            else:
                self.pause_recording = True
        elif q == ord('m'):  # Switch between Manipulation and Navigation modes
            if self.manipulation_mode:
                self.left_down = False
                self.middle_down = False
                self.right_down = False
                self.manipulation_mode = False
            else:
                self.left_down = False
                self.middle_down = False
                self.right_down = False
                self.manipulation_mode = True
        elif q == ord('l'):
            # reset
            self.reset()
        if self.recording and not self.pause_recording:
            cv2.imwrite(os.path.join(self.video_folder, '{:05d}.png'.format(self.frame_idx)),
                        (frame * 255).astype(np.uint8))
            self.frame_idx += 1

        if not self.renderer is None:
            frames = self.renderer.render_robot_cameras(modes=('rgb'))
            if len(frames) > 0:
                frame = cv2.cvtColor(np.concatenate(
                    frames, axis=1), cv2.COLOR_RGB2BGR)
                cv2.imshow('RobotView', frame)


if __name__ == '__main__':
    viewer = Viewer()
    while True:
        viewer.update()
