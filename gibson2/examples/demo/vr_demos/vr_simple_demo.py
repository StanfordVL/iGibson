""" This is a VR demo in a simple scene consisting of some objects to interact with, and space to move around.
Can be used to verify everything is working in VR, and iterate on current VR designs.
"""

import numpy as np
import os
import pybullet as p
import pybullet_data
import time

import gibson2
from gibson2.render.mesh_renderer.mesh_renderer_cpu import MeshRendererSettings
from gibson2.render.mesh_renderer.mesh_renderer_vr import VrSettings
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.articulated_object import ArticulatedObject
from gibson2.robots.behavior_robot import BehaviorRobot
from gibson2.simulator import Simulator
from gibson2 import assets_path
import cv2
from gibson2.utils.mesh_util import quat2rotmat, xyzw2wxyz

# HDR files for PBR rendering
hdr_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_02.hdr')
hdr_texture2 = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'probe_03.hdr')
light_modulation_map_filename = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'Rs_int', 'layout', 'floor_lighttype_0.png')
background_texture = os.path.join(
    gibson2.ig_dataset_path, 'scenes', 'background', 'urban_street_01.jpg')

def get_push_point(bvr_robot):
    eye_pos, eye_orn = bvr_robot.parts['eye'].get_position_orientation()
    mat = quat2rotmat(xyzw2wxyz(eye_orn))[:3, :3]
    view_direction = mat.dot(np.array([1, 0, 0]))
    res = p.rayTest(eye_pos, eye_pos + view_direction * 3)
    hit_pos = None
    if len(res) > 0 and res[0][0] != -1:
        # there is hit
        object_id, link_id, _, hit_pos, hit_normal = res[0]

    return hit_pos



def main():
    # VR rendering settings
    vr_rendering_settings = MeshRendererSettings(optimized=True,
                                                fullscreen=False,
                                                env_texture_filename=hdr_texture,
                                                env_texture_filename2=hdr_texture2,
                                                env_texture_filename3=background_texture,
                                                light_modulation_map_filename=light_modulation_map_filename,
                                                enable_shadow=True,
                                                enable_pbr=True,
                                                msaa=True,
                                                light_dimming_factor=1.0)
    s = Simulator(mode='iggui',
                 rendering_settings=vr_rendering_settings,
                 vr_settings=VrSettings(use_vr=True),
                 image_width=1024,
                 image_height=1024)

    scene = InteractiveIndoorScene('Rs_int', load_object_categories=['walls', 'floors', 'ceilings'], load_room_types=['kitchen'])
    s.import_ig_scene(scene)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    objects = [
        ("jenga/jenga.urdf", (1.300000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (1.200000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (1.100000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (1.000000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (0.900000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("jenga/jenga.urdf", (0.800000, -0.700000, 0.750000), (0.000000, 0.707107, 0.000000,
                0.707107)),
        ("table/table.urdf", (1.000000, -0.200000, 0.000000), (0.000000, 0.000000, 0.707107,
                0.707107)),
        ("duck_vhacd.urdf", (1.050000, -0.500000, 0.700000), (0.000000, 0.000000, 0.707107,
                0.707107)),
        ("duck_vhacd.urdf", (0.950000, -0.100000, 0.700000), (0.000000, 0.000000, 0.707107,
                0.707107)),
        ("sphere_small.urdf", (0.850000, -0.400000, 0.700000), (0.000000, 0.000000, 0.707107,
                0.707107)),
        ("duck_vhacd.urdf", (0.850000, -0.400000, 1.00000), (0.000000, 0.000000, 0.707107,
                0.707107)),
    ]

    for item in objects:
        fpath = item[0]
        pos = item[1]
        orn = item[2]
        item_ob = ArticulatedObject(fpath, scale=1)
        s.import_object(item_ob, use_pbr=False, use_pbr_mapping=False)
        item_ob.set_position(pos)
        item_ob.set_orientation(orn)

    obj = ArticulatedObject(os.path.join(gibson2.ig_dataset_path, 'objects',
        'basket', 'e3bae8da192ab3d4a17ae19fa77775ff', 'e3bae8da192ab3d4a17ae19fa77775ff.urdf'),
                            scale=2)
    s.import_object(obj)
    obj.set_position_orientation([1.1, 0.300000, 1.0], [0, 0, 0, 1])

    bvr_robot = BehaviorRobot(s, use_tracked_body_override=True, show_visual_head=True, use_ghost_hands=False)
    s.import_behavior_robot(bvr_robot)
    s.register_main_vr_robot(bvr_robot)

    max_num_steps = 100

    # bvr_robot.parts['body'].set_position_orientation(
    #     [0, 0, 1.5], [0, 0, 0, 1]
    # )
    # bvr_robot.parts['left_hand'].set_position_orientation(
    #     [0, 0.2, 1.0], [0.5, 0.5, -0.5, 0.5],
    # )
    # bvr_robot.parts['right_hand'].set_position_orientation(
    #     [0, -0.2, 1.0], [-0.5, 0.5, 0.5, 0.5]
    # )
    # bvr_robot.parts['eye'].set_position_orientation(
    #     [0, 0, 1.5], [0, 0, 0, 1]
    # )

    step = 0
    move_dist = 0.3
    rotate_angle = 0.3
    current_pitch = 0
    pitch_angle = 0.5

    action = np.zeros((28,))
    action[19] = 1
    action[27] = 1
    for _ in range(10):
        bvr_robot.set_position_orientation([0, 0, 0.2], [0, 0, 0, 1])
        bvr_robot.update(action)
        s.step()

    # Main simulation loop
    while True:
        user_input = s.step()
        action = np.zeros((28,))


        if user_input == ord('i'):
            action[0] = 0.01
            action[19] = 1
            action[27] = 1
            original_pos = bvr_robot.parts['body'].get_position()
            for _ in range(max_num_steps):
                bvr_robot.update(action)
                s.step()
                if np.linalg.norm(np.array(bvr_robot.parts['body'].get_position()) - np.array(original_pos)) > move_dist:
                    break

        elif user_input == ord('k'):
            action[0] = -0.01
            action[19] = 1
            action[27] = 1
            original_pos = bvr_robot.parts['body'].get_position()
            for _ in range(max_num_steps):
                bvr_robot.update(action)
                s.step()
                if np.linalg.norm(np.array(bvr_robot.parts['body'].get_position()) - np.array(original_pos)) > move_dist:
                    break

        elif user_input == ord('j'):
            action[5] = 0.01
            action[19] = 1
            action[27] = 1
            orn = p.getEulerFromQuaternion(bvr_robot.parts['body'].get_orientation())
            for _ in range(max_num_steps):
                bvr_robot.update(action)
                s.step()
                d = np.abs(orn[2] - p.getEulerFromQuaternion(bvr_robot.parts['body'].get_orientation())[2])
                if d > np.pi / 2:
                    d = np.pi - d
                if  d > rotate_angle:
                    break


        elif user_input == ord('l'):
            action[5] = -0.01
            action[19] = 1
            action[27] = 1
            orn = p.getEulerFromQuaternion(bvr_robot.parts['body'].get_orientation())
            for _ in range(max_num_steps):
                bvr_robot.update(action)
                s.step()
                d = np.abs(orn[2] - p.getEulerFromQuaternion(bvr_robot.parts['body'].get_orientation())[2])
                if d > np.pi / 2:
                    d = np.pi - d
                if d > rotate_angle:
                    break

        elif user_input == ord('o'):
            hit_pos = get_push_point(bvr_robot)
            if hit_pos is not None:
                push_vector = np.array(hit_pos) - np.array(bvr_robot.parts['right_hand'].get_position())
                action[20:23] = push_vector / 100

            for _ in range(max_num_steps):
                bvr_robot.update(action)
                s.step()
                if len(p.getContactPoints(bodyA=bvr_robot.parts['right_hand'].body_id)) > 0:
                    break

        elif user_input == ord('y'):
            current_pitch += 1
            if current_pitch > 1:
                current_pitch = 1
            else:
                action[10] = 0.01
                for _ in range(max_num_steps):
                    bvr_robot.update(action)
                    s.step()
                    d = np.abs(p.getEulerFromQuaternion(bvr_robot.parts['eye'].get_orientation())[1] - pitch_angle * current_pitch)
                    print(d)
                    if d > np.pi / 2:
                        d = np.pi - d
                    if d < 0.01:
                        break

        elif user_input == ord('u'):
            current_pitch -= 1
            if current_pitch < -1:
                current_pitch = -1
            else:
                action[10] = -0.01
                for _ in range(max_num_steps):
                    bvr_robot.update(action)
                    s.step()
                    d = np.abs(p.getEulerFromQuaternion(bvr_robot.parts['eye'].get_orientation())[1] - pitch_angle * current_pitch)
                    if d > np.pi / 2:
                        d = np.pi - d
                    if d < 0.01:
                        break

        elif user_input == ord('p'):
            bvr_robot.set_position_orientation(*bvr_robot.parts['body'].get_position_orientation())
            print(bvr_robot.parts['body'].get_position_orientation())

        rgb = bvr_robot.render_camera_image()[0]
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imshow('robot', rgb)

        action = np.zeros((28,))
        bvr_robot.update(action)

        # Update VR agent using action data from simulator
        #bvr_robot.update(s.gen_vr_robot_action())

    s.disconnect()

if __name__ == '__main__':
    main()