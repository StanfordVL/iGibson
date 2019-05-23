import cv2
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer


class Viewer:
    def __init__(self):
        self.px = 0
        self.py = 0
        self._mouse_ix, self._mouse_iy = -1, -1
        self.down = False
        self.view_direction = np.array([1, 0, 0])

        cv2.namedWindow('test')
        cv2.namedWindow('robots')
        cv2.setMouseCallback('test', self.change_dir)

        self.renderer = None

    def change_dir(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self._mouse_ix, self._mouse_iy = x, y
            self.down = True
        if event == cv2.EVENT_MOUSEMOVE:
            if self.down:
                dx = (x - self._mouse_ix) / 100.0
                dy = (y - self._mouse_iy) / 100.0
                self._mouse_ix = x
                self._mouse_iy = y
                r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0,
                                                                        np.cos(dy)]])
                r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx),
                                                                np.cos(-dx), 0], [0, 0, 1]])
                self.view_direction = r1.dot(r2).dot(self.view_direction)
        elif event == cv2.EVENT_LBUTTONUP:
            self.down = False

    def update(self):
        camera_pose = np.array([self.px, self.py, 1.2])
        if not self.renderer is None:
            self.renderer.set_camera(camera_pose, camera_pose + self.view_direction, [0, 0, 1])

        if not self.renderer is None:
            frame = cv2.cvtColor(np.concatenate(self.renderer.render(modes=("rgb")), axis=1),
                                 cv2.COLOR_RGB2BGR)
        else:
            frame = np.zeros((300, 300, 3)).astype(np.uint8)
        #cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
        cv2.putText(frame, "px {:1.1f} py {:1.1f}".format(self.px, self.py), (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "[{:1.1f} {:1.1f} {:1.1f}]".format(*self.view_direction), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('test', frame)

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

        if not self.renderer is None:
            frames = self.renderer.render_robot_cameras()
            if len(frames) > 0:
                frame = cv2.cvtColor(np.concatenate(frames, axis=1), cv2.COLOR_RGB2BGR)
                cv2.imshow('robots', frame)


if __name__ == '__main__':
    viewer = Viewer()
    while True:
        viewer.update()
