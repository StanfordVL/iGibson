import os
import pybullet as p
import numpy as np
from transforms3d.euler import euler2quat
from igibson.utils.utils import quatToXYZW

from igibson.external.pybullet_tools.utils import stable_z_on_aabb
from igibson.external.pybullet_tools.utils import get_center_extent
from igibson.simulator import Simulator
from igibson.objects.articulated_object import ArticulatedObject
from PIL import Image
import json


def main():
    step_per_sec = 100
    num_directions = 12
    obj_count = 0
    root_dir = '/cvgl2/u/chengshu/ig_dataset_v5/objects'

    s = Simulator(mode='headless',
                  image_width=512,
                  image_height=512,
                  physics_timestep=1 / float(step_per_sec))
    p.setGravity(0.0, 0.0, 0.0)

    for obj_class_dir in sorted(os.listdir(root_dir)):
        obj_class_dir = os.path.join(root_dir, obj_class_dir)
        for obj_inst_dir in os.listdir(obj_class_dir):
            obj_inst_name = obj_inst_dir
            urdf_path = obj_inst_name + '.urdf'
            obj_inst_dir = os.path.join(obj_class_dir, obj_inst_dir)
            urdf_path = os.path.join(obj_inst_dir, urdf_path)

            obj = ArticulatedObject(urdf_path)
            s.import_object(obj)

            with open(os.path.join(obj_inst_dir, 'misc/bbox.json'), 'r') as bbox_file:
                bbox_data = json.load(bbox_file)
                bbox_max = np.array(bbox_data['max'])
                bbox_min = np.array(bbox_data['min'])
            offset = -(bbox_max + bbox_min) / 2.0

            z = stable_z_on_aabb(obj.body_id, [[0, 0, 0], [0, 0, 0]])

            obj.set_position([offset[0], offset[1], z])
            _, extent = get_center_extent(obj.body_id)

            max_half_extent = max(extent) / 2.0
            px = max_half_extent * 3.0
            py = 0.0
            pz = extent[2] / 2.0
            camera_pose = np.array([px, py, pz])

            s.renderer.set_camera(camera_pose,
                                  [0, 0, pz],
                                  [0, 0, 1])

            num_joints = p.getNumJoints(obj.body_id)
            if num_joints == 0:
                s.reload()
                continue

            # collect joint low/high limit
            joint_low = []
            joint_high = []
            for j in range(num_joints):
                j_low, j_high = p.getJointInfo(obj.body_id, j)[8:10]
                joint_low.append(j_low)
                joint_high.append(j_high)

            # set joints to their lowest limits
            for j, j_low in zip(range(num_joints), joint_low):
                p.resetJointState(
                    obj.body_id, j, targetValue=j_low, targetVelocity=0.0)
            s.sync()

            # render the images
            joint_low_imgs = []
            for i in range(num_directions):
                yaw = np.pi * 2.0 / num_directions * i
                obj.set_orientation(quatToXYZW(
                    euler2quat(0.0, 0.0, yaw), 'wxyz'))
                s.sync()
                rgb, three_d = s.renderer.render(modes=('rgb', '3d'))
                depth = -three_d[:, :, 2]
                rgb[depth == 0] = 1.0
                joint_low_imgs.append(Image.fromarray(
                    (rgb[:, :, :3] * 255).astype(np.uint8)))

            # set joints to their highest limits
            for j, j_high in zip(range(num_joints), joint_high):
                p.resetJointState(
                    obj.body_id, j, targetValue=j_high, targetVelocity=0.0)
            s.sync()

            # render the images
            joint_high_imgs = []
            for i in range(num_directions):
                yaw = np.pi * 2.0 / num_directions * i
                obj.set_orientation(quatToXYZW(
                    euler2quat(0.0, 0.0, yaw), 'wxyz'))
                s.sync()
                rgb, three_d = s.renderer.render(modes=('rgb', '3d'))
                depth = -three_d[:, :, 2]
                rgb[depth == 0] = 1.0
                joint_high_imgs.append(Image.fromarray(
                    (rgb[:, :, :3] * 255).astype(np.uint8)))

            # concatenate the images
            imgs = []
            for im1, im2 in zip(joint_low_imgs, joint_high_imgs):
                dst = Image.new('RGB', (im1.width + im2.width, im1.height))
                dst.paste(im1, (0, 0))
                dst.paste(im2, (im1.width, 0))
                imgs.append(dst)
            gif_path = '{}/visualizations/{}_joint_limit.gif'.format(
                obj_inst_dir, obj_inst_name)

            # save the gif
            imgs[0].save(gif_path,
                         save_all=True,
                         append_images=imgs[1:],
                         optimize=True,
                         duration=200,
                         loop=0)

            s.reload()
            obj_count += 1
            print(obj_count, gif_path)


if __name__ == "__main__":
    main()
