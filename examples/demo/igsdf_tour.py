import gibson2
from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.profiler import Profiler
from gibson2.utils.assets_utils import get_ig_scene_path
import argparse
import os
from PIL import Image
import numpy as np
import subprocess
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, 
                        help='Name of the scene in the iG Dataset')
    parser.add_argument('--save_dir', type=str, help='Directory to save the frames.')
    parser.add_argument('--seed', type=int, default=15, 
                        help='Random seed.')
    parser.add_argument('--domain_rand', dest='domain_rand',
                        action='store_true')
    parser.add_argument('--domain_rand_interval', dest='domain_rand_interval',
                        type=int, default=50)
    parser.add_argument('--object_rand', dest='object_rand',
                        action='store_true')
    args = parser.parse_args()

    # hdr_texture1 = os.path.join(
                 # gibson2.ig_dataset_path, 'scenes', 'background', 'photo_studio_01_2k.hdr')
    hdr_texture1 = os.path.join(
                 gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
    hdr_texture2 = os.path.join(
                 gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
    light_modulation_map_filename = os.path.join(
                 get_ig_scene_path(args.scene), 'layout', 'floor_lighttype_0.png')

    background_texture = os.path.join(
                gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

    settings = MeshRendererSettings(
               env_texture_filename=hdr_texture1,
               env_texture_filename2=hdr_texture2,
               env_texture_filename3=background_texture,
               enable_shadow=True, msaa=True,
               light_dimming_factor=1.2)

    s = Simulator(mode='headless', 
            image_width=1080, image_height=720, 
            vertical_fov=60, rendering_settings=settings
            )

    random.seed(args.seed)
    scene = InteractiveIndoorScene(
            args.scene, texture_randomization=args.domain_rand,
            object_randomization=args.object_rand)

    s.import_ig_scene(scene)
    s.renderer.use_pbr(use_pbr=True, use_pbr_mapping=True)
    s.renderer.set_light_position_direction([0,0,10], [0,0,0])

    traj_path = os.path.join(get_ig_scene_path(args.scene), 'misc', 
                             'tour_cam_trajectory.txt')
    save_dir = os.path.join(get_ig_scene_path(args.scene), args.save_dir)
    os.makedirs(save_dir, exist_ok=True)
    with open(traj_path, 'r') as fp:
        points = [l.rstrip().split(',') for l in fp.readlines()]

    s.sync()
    for i in range(len(points)):
        if args.domain_rand and i % args.domain_rand_interval == 0:
            scene.randomize_texture()
        x,y,dir_x,dir_y = [float(p) for p in points[i]]
        z = 1.7
        tar_x = x+dir_x
        tar_y = y+dir_y
        tar_z = 1.4
        # cam_loc = np.array([x, y, z])
        s.renderer.set_camera([x, y, z], [tar_x,tar_y,tar_z], [0, 0, 1])

        with Profiler('Render'):
            frame = s.renderer.render(modes=('rgb'))
        img = Image.fromarray((
                255*np.concatenate(frame, axis=1)[:,:,:3]).astype(np.uint8))
        img.save(os.path.join(save_dir, '{:05d}.png'.format(i)))
    
    cmd = 'ffmpeg -i {s}/%5d.png -y -c:a copy -c:v libx264 -crf 18 -preset veryslow -r 30 {s}/viz.mp4'.format(s=save_dir)
    subprocess.call(cmd, shell=True)
    cmd = 'rm {}/*.png'.format(save_dir)
    subprocess.call(cmd, shell=True)

    s.disconnect()


if __name__ == '__main__':
    main()
