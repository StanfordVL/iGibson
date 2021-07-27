import cv2
import sys
import os
import numpy as np
from igibson.simulator import Simulator
from igibson.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
# from igibson.utils.assets_utils import get_model_path
from igibson.objects.articulated_object import ArticulatedObject
import igibson
from PIL import Image
import pybullet as p
import subprocess

def load_obj_np(filename_obj, normalization=False, texture_size=4, load_texture=False,
                texture_wrapping='REPEAT', use_bilinear=True):
    """Load Wavefront .obj file into numpy array
    This function only supports vertices (v x x x) and faces (f x x x).
    """
    # load vertices
    vertices = []
    with open(filename_obj) as f:
        lines = f.readlines()

    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'v':
            vertices.append([float(v) for v in line.split()[1:4]])
    vertices = np.vstack(vertices).astype(np.float32)

    # load faces
    faces = []
    for line in lines:
        if len(line.split()) == 0:
            continue
        if line.split()[0] == 'f':
            vs = line.split()[1:]
            nv = len(vs)
            v0 = int(vs[0].split('/')[0])
            for i in range(nv - 2):
                v1 = int(vs[i + 1].split('/')[0])
                v2 = int(vs[i + 2].split('/')[0])
                faces.append((v0, v1, v2))
    faces = np.vstack(faces).astype(np.int32) - 1

    # load textures
    textures = None

    assert load_texture is False  # Since I commented out the block below
    # if load_texture:
    #     for line in lines:
    #         if line.startswith('mtllib'):
    #             filename_mtl = os.path.join(os.path.dirname(filename_obj), line.split()[1])
    #             textures = load_textures(filename_obj, filename_mtl, texture_size,
    #                                      texture_wrapping=texture_wrapping,
    #                                      use_bilinear=use_bilinear)
    #     if textures is None:
    #         raise Exception('Failed to load textures.')
    #     textures = textures.cpu().numpy()

    assert normalization is False  # Since I commented out the block below
    # # normalize into a unit cube centered zero
    # if normalization:
    #     vertices -= vertices.min(0)[0][None, :]
    #     vertices /= torch.abs(vertices).max()
    #     vertices *= 2
    #     vertices -= vertices.max(0)[0][None, :] / 2

    if load_texture:
        return vertices, faces, textures
    else:
        return vertices, faces


def main():
    global _mouse_ix, _mouse_iy, down, view_direction

    model_path = sys.argv[1]
    print(model_path)

    model_id = os.path.basename(model_path)
    category = os.path.basename(os.path.dirname(model_path))

    hdr_texture = os.path.join(
                igibson.ig_dataset_path, 'scenes', 'background',
                'photo_studio_01_2k.hdr')
    settings = MeshRendererSettings(env_texture_filename=hdr_texture,
               enable_shadow=True, msaa=True,
               light_dimming_factor=1.5)

    s = Simulator(mode='headless', 
            image_width=1800, image_height=1200, 
            vertical_fov=70, rendering_settings=settings
            )

    s.renderer.set_light_position_direction([0,0,10], [0,0,0])

    s.renderer.load_object('plane/plane_z_up_0.obj', scale=[3,3,3])
    s.renderer.add_instance(0)
    s.renderer.set_pose([0,0,-1.5,1, 0, 0.0, 0.0], -1)


    v = []
    mesh_path = os.path.join(model_path, 'shape/visual')
    for fn in os.listdir(mesh_path):
        if fn.endswith('obj'):
            vertices, faces = load_obj_np(os.path.join(mesh_path, fn))
            v.append(vertices)

    v = np.vstack(v)
    print(v.shape)
    xlen = np.max(v[:,0]) - np.min(v[:,0])
    ylen = np.max(v[:,1]) - np.min(v[:,1])
    zlen = np.max(v[:,2]) - np.min(v[:,2])
    scale = 1.5/(max([xlen, ylen, zlen]))
    center = np.mean(v, axis=0)
    centered_v = v - center

    center = (np.max(v, axis=0) + np.min(v, axis=0)) / 2.

    urdf_path = os.path.join(model_path, '{}.urdf'.format(model_id))
    print(urdf_path)
    obj = ArticulatedObject(filename=urdf_path, scale=scale)
    s.import_object(obj)
    obj.set_position(center)
    s.sync()
    print(s.renderer.visual_objects, s.renderer.instances)

    _mouse_ix, _mouse_iy = -1, -1
    down = False

    theta,r = 0,1.5

    px = r*np.sin(theta)
    py = r*np.cos(theta)
    pz = 1
    camera_pose = np.array([px, py, pz])
    s.renderer.set_camera(camera_pose, [0,0,0], [0, 0, 1])

    num_views = 6 
    save_dir = os.path.join(model_path, 'visualizations')
    for i in range(num_views):
        theta += np.pi*2/(num_views+1)
        obj.set_orientation([0., 0., 1.0, np.cos(theta/2)])
        s.sync()
        with Profiler('Render'):
            frame = s.renderer.render(modes=('rgb'))
        img = Image.fromarray((
                255*np.concatenate(frame, axis=1)[:,:,:3]).astype(np.uint8))
        img.save(os.path.join(save_dir, '{:02d}.png'.format(i)))

    cmd = 'ffmpeg -framerate 2 -i {s}/%2d.png -y -r 16 -c:v libx264 -pix_fmt yuvj420p {s}/{m}.mp4'.format(s=save_dir,m=model_id)
    subprocess.call(cmd, shell=True)
    cmd = 'rm {}/??.png'.format(save_dir)
    subprocess.call(cmd, shell=True)

if __name__ == '__main__':
    main()
