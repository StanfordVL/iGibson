import numpy as np
import time
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import sys
import os
import cv2


def benchmark(render_to_tensor=False, resolution=512, obj_num = 100, optimized = True):
    
    n_frame = 200

    if optimized:
        settings = MeshRendererSettings(msaa=True, optimized=True)
        renderer = MeshRenderer(width=resolution, height=resolution, vertical_fov=90, rendering_settings=settings)
    else:
        settings = MeshRendererSettings(msaa=True, optimized=False)
        renderer = MeshRenderer(width=resolution, height=resolution, vertical_fov=90, rendering_settings=settings)

    renderer.load_object('plane/plane_z_up_0.obj', scale=[3,3,3])
    renderer.add_instance(0)
    renderer.instances[-1].use_pbr = True
    renderer.instances[-1].use_pbr_mapping = True
    renderer.set_pose([0,0,-1.5,1, 0, 0.0, 0.0], -1)

    
    model_path = sys.argv[1]

    px = 1
    py = 1
    pz = 1

    camera_pose = np.array([px, py, pz])
    view_direction = np.array([-1, -1, -1])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    theta = 0
    r = 6
    scale = 1    
    i = 1

    obj_count_x = int(np.sqrt(obj_num))


    for fn in os.listdir(model_path):
        if fn.endswith('obj') and 'processed' in fn:
            renderer.load_object(os.path.join(model_path, fn), scale=[scale, scale, scale])
            for obj_i in range(obj_count_x):
                for obj_j in range(obj_count_x):        
                    renderer.add_instance(i)
                    renderer.set_pose([obj_i-obj_count_x/2., obj_j-obj_count_x/2.,0,0.7071067690849304, 0.7071067690849304, 0.0, 0.0], -1)
                    renderer.instances[-1].use_pbr = True
                    renderer.instances[-1].use_pbr_mapping = True

            i += 1
            

    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)

    start = time.time()
    for i in range(n_frame):
        px = r*np.sin(theta)
        py = r*np.cos(theta)
        theta += 0.01
        camera_pose = np.array([px, py, pz])
        renderer.set_camera(camera_pose, [0,0,0], [0, 0, 1])

        frame = renderer.render(modes=('rgb', 'normal'))
        #print(frame)
        cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
    elapsed = time.time()-start
    print('{} fps'.format(n_frame/elapsed))
    return obj_num, n_frame/elapsed

def main():
    #benchmark(render_to_tensor=True, resolution=512)
    results = []
    
    for obj_num in [item **2 for item in [10]]:
        res = benchmark(render_to_tensor=False, resolution=512, obj_num=obj_num, optimized = True)

if __name__ == '__main__':
    main()
