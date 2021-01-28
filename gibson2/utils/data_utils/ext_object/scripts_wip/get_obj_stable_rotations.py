'''
Credit: Andrey Kurenkov
'''

import os
import trimesh
import argparse
import numpy as np
import json
import math
from pyquaternion import Quaternion
import pybullet as p
import matplotlib.pyplot as plt

'''
Analyzes a model for possible ways to place it flat on a surface.
Use by running without --ask_probs or --save_json to see rotations, and then with to save them with probabilities.
'''
def viz_transform(body_id, mesh, transform):
    quat = Quaternion(matrix=transform)
    r = quat.real
    v = quat.vector
    p.resetBasePositionAndOrientation(
        body_id,
        [0, 0, mesh.extents[2]/2.0],
        [quat.vector[0], quat.vector[1], quat.vector[2], quat.real])
    return r, v


def main(args):
    p.connect(p.GUI)
    metadata_file = 'data/ig_dataset/objects/%s/%s/misc/metadata.json'%(args.object_cat, args.object_id)
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    obj_dir = 'data/ig_dataset/objects/%s/%s/shape/visual'%(args.object_cat, args.object_id)
    obj_name = [f for f in os.listdir(obj_dir) if 'obj' in f][0]
    object_file = os.path.join(obj_dir,obj_name)

    mesh = trimesh.load(object_file, force='mesh')
    poses = trimesh.poses.compute_stable_poses(
        mesh, n_samples=5, threshold=args.threshold)

    visualShapeId = p.createVisualShape(shapeType=p.GEOM_MESH,
                                        fileName=object_file)
    collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_MESH,
                                              fileName=object_file)
    body_id = p.createMultiBody(baseCollisionShapeIndex=collisionShapeId,
                                baseVisualShapeIndex=visualShapeId)

    info_dict = {}
    aabb = p.getAABB(body_id)
    print('Showing all stable placement rotations:')
    dicts = []

    for i in range(len(poses[0])):
        rotation_dict = {}
        transform = poses[0][i]
        r, v = viz_transform(body_id, mesh, transform)
        prob = 0
        if args.save_json:
            inp = input(
                'Enter probability of rotation, or +/- to rotate about Z:')
            if inp == '':
                continue
            while inp[0] == '+' or inp[0] == '-':
                rot_num = float(inp[1:])
                if inp[0] == '-':
                    rot_num *= -1
                z_rot = np.array([[math.cos(math.pi*rot_num), -math.sin(math.pi*rot_num), 0.0, 0.0],
                                  [math.sin(math.pi*rot_num),
                                   math.cos(math.pi*rot_num), 0.0, 0.0],
                                  [0.0, 0.0, 1.0, 0.0],
                                  [0.0, 0.0, 0.0, 1.0]])
                transform = np.matmul(z_rot, transform)
                r, v = viz_transform(body_id, mesh, transform)
                inp = input(
                    'Enter probability of rotation, or +/- to rotate about Z:')
            prob = float(inp)
            variation = float(input('Enter variation about Z (0-1):'))
            rotation_dict['prob'] = prob
            rotation_dict['variation'] = variation
        else:
            skip = input('Rotation %d: (press enter to continue)' % (i+1))
        rotation_dict['rotation'] = [
            float(v[0]), float(v[1]), float(v[2]), float(r)]
        aabb = p.getAABB(body_id)
        size = [aabb[1][0] - aabb[0][0], aabb[1]
                [1] - aabb[0][1], aabb[1][2] - aabb[0][2]]
        print('Bounding box size=%s' % str(size))
        rotation_dict['size'] = size
        if prob > 0:
            dicts.append(rotation_dict)

    print('Summary:')
    for d in dicts:
        print(d)

    if args.save_json:
        metadata['orientations'] = dicts

        with open(metadata_file, 'w') as f:
            documents = json.dump(metadata, f)

    p.removeBody(body_id)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyze objects that can be placed in a container for their plausible rotations.')
    parser.add_argument('object_cat', type=str)
    parser.add_argument('object_id', type=str)
    parser.add_argument('--save_json', action='store_true',
                        help='Whether to ask for and save orientations and probabilities to json.')
    parser.add_argument('--threshold', type=float, default=0.03,
                        help='Threshold for including orientations or not.')
    args = parser.parse_args()
    main(args)
