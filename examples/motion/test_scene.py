from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import *
import os
from PIL import Image
from gibson2.core.render.mesh_renderer.glutils.meshutil import get_params
if __name__ == "__main__":
    renderer = MeshRenderer(width=1280, height=720)
    

    models = os.listdir('/home/fei/Development/optix/Optix-PathTracer-RTRT/src/data/hyperion')    
    print(models)

    for i in range(len(models)):
        renderer.load_object(os.path.join('/home/fei/Development/optix/Optix-PathTracer-RTRT/src/data/hyperion', models[i]))
        renderer.add_instance(i)

    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)
    camera_pose = np.array([-12.4616, 15.8139, 18.4849])
    view_direction = np.array([ -8.90342e-07, 1.53424, 1.13471e-06]) - camera_pose
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 1, 0])
    renderer.set_fov(35)

    print(renderer.get_camera())

    Image.fromarray((255*renderer.render(modes=('rgb'))[0][:,:,:3]).astype(np.uint8)).save('test_rgb.png')
    Image.fromarray((255*renderer.render(modes=('normal'))[0][:,:,:3]).astype(np.uint8)).save('test_normal.png')

    renderer.release()
