import cv2
import sys
import os
import numpy as np
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.profiler import Profiler
from igibson.utils.assets_utils import get_scene_path


def main():
    global _mouse_ix, _mouse_iy, down, view_direction

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path('Rs'), 'mesh_z_up.obj')

    renderer = MeshRenderer(width=512, height=512)
    renderer.load_object(model_path)

    renderer.add_instance(0)
    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)

    px = 0
    py = 0.2

    camera_pose = np.array([px, py, 0.5])
    view_direction = np.array([0, -1, -1])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)

    _mouse_ix, _mouse_iy = -1, -1
    down = False

    def change_dir(event, x, y, flags, param):
        global _mouse_ix, _mouse_iy, down, view_direction
        if event == cv2.EVENT_LBUTTONDOWN:
            _mouse_ix, _mouse_iy = x, y
            down = True
        if event == cv2.EVENT_MOUSEMOVE:
            if down:
                dx = (x - _mouse_ix) / 100.0
                dy = (y - _mouse_iy) / 100.0
                _mouse_ix = x
                _mouse_iy = y
                r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [
                              0, 1, 0], [-np.sin(dy), 0, np.cos(dy)]])
                r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0],
                               [np.sin(-dx), np.cos(-dx), 0], [0, 0, 1]])
                view_direction = r1.dot(r2).dot(view_direction)
        elif event == cv2.EVENT_LBUTTONUP:
            down = False

    cv2.namedWindow('test')
    cv2.setMouseCallback('test', change_dir)

    while True:
        with Profiler('Render'):
            frame = renderer.render(modes=('rgb'))
        cv2.imshow('test', cv2.cvtColor(
            np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
        q = cv2.waitKey(1)
        if q == ord('w'):
            px += 0.01
        elif q == ord('s'):
            px -= 0.01
        elif q == ord('a'):
            py += 0.01
        elif q == ord('d'):
            py -= 0.01
        elif q == ord('q'):
            break
        camera_pose = np.array([px, py, 0.5])
        renderer.set_camera(camera_pose, camera_pose +
                            view_direction, [0, 0, 1])

    renderer.release()


if __name__ == '__main__':
    main()
