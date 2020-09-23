import cv2
import sys
import os
import numpy as np
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRenderer
from gibson2.render.profiler import Profiler
from gibson2.utils.assets_utils import get_scene_path
from PIL import Image

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

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_scene_path('Rs'), 'mesh_z_up.obj')

    renderer = MeshRenderer(width=512, height=512, msaa=True, enable_shadow=True, vertical_fov=90)
    renderer.set_light_position_direction([0,0,1.5], [0,0,0])

    renderer.load_object('plane/plane_z_up_0.obj', scale=[3,3,3])
    renderer.add_instance(0)
    renderer.set_pose([0,0,-1.5,1, 0, 0.0, 0.0], -1)

    i = 1

    v = []
    for fn in os.listdir(model_path):
        if fn.endswith('obj'):
            vertices, faces = load_obj_np(os.path.join(model_path, fn))
            v.append(vertices)

    v = np.vstack(v)
    print(v.shape)
    xlen = np.max(v[:,0]) - np.min(v[:,0])
    ylen = np.max(v[:,1]) - np.min(v[:,1])
    scale = 1.0/(max(xlen, ylen))

    for fn in os.listdir(model_path):
        if fn.endswith('obj'):
            renderer.load_object(os.path.join(model_path, fn), scale=[scale, scale, scale])
            renderer.add_instance(i)
            i += 1
            renderer.instances[-1].use_pbr = True
            renderer.instances[-1].use_pbr_mapping = True
            renderer.instances[-1].metalness = 1
            renderer.instances[-1].roughness = 0.1
            

    print(renderer.visual_objects, renderer.instances)
    print(renderer.materials_mapping, renderer.mesh_materials)
    

    px = 1
    py = 1
    pz = 1

    camera_pose = np.array([px, py, pz])
    view_direction = np.array([-1, -1, -1])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])

    _mouse_ix, _mouse_iy = -1, -1
    down = False

    # def change_dir(event, x, y, flags, param):
    #     global _mouse_ix, _mouse_iy, down, view_direction
    #     if event == cv2.EVENT_LBUTTONDOWN:
    #         _mouse_ix, _mouse_iy = x, y
    #         down = True
    #     if event == cv2.EVENT_MOUSEMOVE:
    #         if down:
    #             dx = (x - _mouse_ix) / 100.0
    #             dy = (y - _mouse_iy) / 100.0
    #             _mouse_ix = x
    #             _mouse_iy = y
    #             r1 = np.array([[np.cos(dy), 0, np.sin(dy)], [0, 1, 0], [-np.sin(dy), 0, np.cos(dy)]])
    #             r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx), np.cos(-dx), 0], [0, 0, 1]])
    #             view_direction = r1.dot(r2).dot(view_direction)
    #     elif event == cv2.EVENT_LBUTTONUP:
    #         down = False

    # cv2.namedWindow('test')
    # cv2.setMouseCallback('test', change_dir)

    theta = 0
    r = 1.5
    imgs = []
    for i in range(60):
        theta += np.pi*2/60
        renderer.set_pose([0,0,-1.5,np.cos(-theta/2), 0, 0.0, np.sin(-theta/2)], 0)
        with Profiler('Render'):
            frame = renderer.render(modes=('rgb'))
        cv2.imshow('test', cv2.cvtColor(np.concatenate(frame, axis=1), cv2.COLOR_RGB2BGR))

        imgs.append(Image.fromarray((255*np.concatenate(frame, axis=1)[:,:,:3]).astype(np.uint8)))

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

        px = r*np.sin(theta)
        py = r*np.cos(theta)
        camera_pose = np.array([px, py, pz])
        renderer.set_camera(camera_pose, [0,0,0], [0, 0, 1])

    renderer.release()
    imgs[0].save('{}.gif'.format('/data2/gifs/' + model_path.replace('/', '_')),
                   save_all=True, append_images=imgs[1:], optimize=False, duration=40, loop=0)

if __name__ == '__main__':
    main()