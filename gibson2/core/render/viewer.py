import cv2
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer


class Viewer:
    def __init__(self, 
                 initial_pos = [0,0,1], 
                 initial_view_direction = [1,0,0], 
                 initial_up = [0,0,1],
                 mujoco_env = None,
                 render_static_cam = False):

        self.px = initial_pos[0]#-0.5
        self.py = initial_pos[1]#0.3
        self.pz = initial_pos[2]#1.0
        self._mouse_ix, self._mouse_iy = -1, -1
        self.left_down = False
        self.middle_down = False
        self.right_down = False
        self.view_direction = np.array(initial_view_direction)#np.array([0.9, -0.4, 0.1])
        self.up = initial_up
        self.renderer = None
        self.static_cam_idx = 0
        self.render_static_cam = render_static_cam

        cv2.namedWindow('ExternalView')
        cv2.moveWindow("ExternalView", 0,0);

        if not render_static_cam:
            cv2.namedWindow('RobotView')
        else:
            cv2.namedWindow('StaticCamView')

        cv2.setMouseCallback('ExternalView', self.change_dir)

        self.mujoco_env = mujoco_env

    def change_dir(self, event, x, y, flags, param):
        
        if flags == cv2.EVENT_FLAG_LBUTTON + cv2.EVENT_FLAG_CTRLKEY and not self.right_down: # Only once
            self._mouse_ix, self._mouse_iy = x, y
            self.right_down = True
        elif event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_ix, self._mouse_iy = x, y
            self.left_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.left_down = False
            self.right_down = False
        elif event == cv2.EVENT_MBUTTONDOWN:
            self._mouse_ix, self._mouse_iy = x, y
            self.middle_down = True
        elif event == cv2.EVENT_MBUTTONUP:
            self.middle_down = False

        if event == cv2.EVENT_MOUSEMOVE:
            if self.left_down:
                dx = (x - self._mouse_ix) / 100.0
                dy = (y - self._mouse_iy) / 100.0
                self._mouse_ix = x
                self._mouse_iy = y
                r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0,
                                                                        np.cos(dy)]])
                r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx),
                                                                np.cos(-dx), 0], [0, 0, 1]])
                self.view_direction = r1.dot(r2).dot(self.view_direction)
            elif self.middle_down:
                d_vd = (y - self._mouse_iy) / 100.0
                self._mouse_iy = y

                motion_along_vd = d_vd*self.view_direction
                self.px += motion_along_vd[0]
                self.py += motion_along_vd[1]
                self.pz += motion_along_vd[2]
            elif self.right_down:

                zz = self.view_direction/np.linalg.norm(self.view_direction)
                xx = np.cross(zz, np.array([0,0,1]))
                xx = xx/np.linalg.norm(xx)
                yy = np.cross(xx, zz)
                motion_along_vx = -((x - self._mouse_ix) / 100.0)*xx
                motion_along_vy = ((y - self._mouse_iy) / 100.0)*yy
                self._mouse_ix = x
                self._mouse_iy = y

                self.px += (motion_along_vx[0] + motion_along_vy[0])
                self.py += (motion_along_vx[1] + motion_along_vy[1])
                self.pz += (motion_along_vx[2] + motion_along_vy[2])
            

    def update(self):
        camera_pose = np.array([self.px, self.py, self.pz])
        if not self.renderer is None:
            self.renderer.set_camera(camera_pose, camera_pose + self.view_direction, self.up)

        if not self.renderer is None:
            frame = cv2.cvtColor(np.concatenate(self.renderer.render(modes=('rgb')), axis=1),
                                 cv2.COLOR_RGB2BGR)
        else:
            frame = np.zeros((300, 300, 3)).astype(np.uint8)
        #cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
        cv2.putText(frame, "px {:1.1f} py {:1.1f} pz {:1.1f}".format(self.px, self.py, self.pz), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "[{:1.1f} {:1.1f} {:1.1f}]".format(*self.view_direction), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('ExternalView', frame)

        q = cv2.waitKey(1)
        if q == ord('w'):
            self.px += 0.05
        elif q == ord('s'):
            self.px -= 0.05
        elif q == ord('a'):
            self.py += 0.05
        elif q == ord('d'):
            self.py -= 0.05
        elif q == ord('q'):
            exit()
        elif q == ord('\t'):
            self.static_cam_idx = (self.static_cam_idx+1)%len(self.renderer.get_static_camera_names())

        if not self.renderer is None:
            if not self.render_static_cam:
                frames = self.renderer.render_robot_cameras(modes=('rgb'))
                if len(frames) > 0:
                    frame = cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
                    cv2.imshow('RobotView', frame)
            else:
                frames = self.renderer.render_static_camera(self.renderer.get_static_camera_names()[self.static_cam_idx], modes=('rgb'))
                if len(frames) > 0:
                    frame = cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
                    cv2.imshow('StaticCamView', frame)

    def set_camera(self, camera_id=0):
        self.static_cam_idx = camera_id


if __name__ == '__main__':
    viewer = Viewer()
    while True:
        viewer.update()
