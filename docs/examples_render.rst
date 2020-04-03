(Deprecated) Examples of renderer
=================================

Use the mesh renderer
---------------------

You can use the mesh renderer to render an iGibson mesh within lines of code:

.. code-block:: python

    import cv2
    import sys
    import numpy as np
    from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer

    if __name__ == '__main__':
        model_path = sys.argv[1]
        renderer = MeshRenderer(width=512, height=512)
        renderer.load_object(model_path)
        renderer.add_instance(0)

        print(renderer.visual_objects, renderer.instances)
        print(renderer.materials_mapping, renderer.mesh_materials)
        camera_pose = np.array([0, 0, 1.2])
        view_direction = np.array([1, 0, 0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        renderer.set_fov(90)
        frame = renderer.render(modes=('rgb', 'normal', '3d'))

Create a FPS style interactive mesh renderer
----------------------------------------------
The code can be found in `examples/demo/mesh_renderer_example.py`.

.. code-block:: python

    import cv2
    import sys
    import numpy as np
    from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import VisualObject, InstanceGroup, MeshRenderer

    if __name__ == '__main__':
        model_path = sys.argv[1]
        renderer = MeshRenderer(width=512, height=512)
        renderer.load_object(model_path)
        renderer.add_instance(0)

        print(renderer.visual_objects, renderer.instances)
        print(renderer.materials_mapping, renderer.mesh_materials)
        camera_pose = np.array([0, 0, 1.2])
        view_direction = np.array([1, 0, 0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        renderer.set_fov(90)

        px = 0
        py = 0

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
                    r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0, np.cos(dy)]])
                    r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx), np.cos(-dx), 0], [0, 0, 1]])
                    view_direction = r1.dot(r2).dot(view_direction)
            elif event == cv2.EVENT_LBUTTONUP:
                down = False


        cv2.namedWindow('test')
        cv2.setMouseCallback('test', change_dir)

        while True:
            frame = renderer.render(modes=('rgb', 'normal', '3d'))
            cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
            q = cv2.waitKey(1)
            if q == ord('w'):
                px += 0.05
            elif q == ord('s'):
                px -= 0.05
            elif q == ord('a'):
                py += 0.05
            elif q == ord('d'):
                py -= 0.05
            elif q == ord('q'):
                break
            camera_pose = np.array([px, py, 1.2])
            renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

        renderer.release()

Sample rendering results with 'Ribera' scene would look like below:

.. image:: images/renderer_example.png
    :width: 600


Use the mesh renderer to render to tensor
--------------------------------------------

You can use iGibson's mesh renderer to render to a pytorch tensor, and it is extremely fast. Pytorch installation is required (otherwise, iGibson's simulator is not dependent on pytorch.)

The code can be found in `examples/demo/mesh_renderer_example.py`.

.. code-block:: python

    import cv2
    import sys
    import numpy as np
    from gibson2.core.render.mesh_renderer.mesh_renderer_tensor import MeshRendererG2G
    from gibson2.core.render.profiler import Profiler
    import matplotlib.pyplot as plt
    import torch

    if __name__ == '__main__':
        model_path = sys.argv[1]
        renderer = MeshRendererG2G(width=512, height=512, device_idx=0)
        renderer.load_object(model_path)
        renderer.add_instance(0)

        print(renderer.visual_objects, renderer.instances)
        print(renderer.materials_mapping, renderer.mesh_materials)
        camera_pose = np.array([0, 0, 1.2])
        view_direction = np.array([1, 0, 0])
        renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
        renderer.set_fov(90)
        for i in range(3000):
            with Profiler('Render'):
                frame = renderer.render(modes=('rgb', 'normal'))

        print(frame)
        img_np = frame[0].flip(0).data.cpu().numpy().reshape(renderer.height, renderer.width, 4)
        normal_np = frame[1].flip(0).data.cpu().numpy().reshape(renderer.height, renderer.width, 4)
        plt.imshow(np.concatenate([img_np, normal_np], axis=1))
        plt.show()

        renderer.release()

On `Ribera` scene, rendering 'rgb' and 'normal' at 512x512 on a GTX 1080ti, a framerate of 1300+ fps can be achieved.
