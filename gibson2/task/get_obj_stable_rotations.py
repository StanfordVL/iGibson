'''
Credit: Andrey Kurenkov 
'''

import os
import trimesh
import argparse
import numpy as np
import yaml
import math
from pyquaternion import Quaternion
from transformations import euler_from_matrix
import pybullet as pb
from gibson2.simulator import Simulator
from gibson2.objects.articulated_object import ArticulatedObject
import matplotlib.pyplot as plt

'''
Analyzes a model for possible ways to place it flat on a surface.
Use by running without --ask_probs or --save_yaml to see rotations, and then with to save them with probabilities.
'''

simulator = Simulator(image_width=640, image_height=640)#, mode='headless')
parser = argparse.ArgumentParser(description='Analyze objects that can be placed in a container.')
parser.add_argument('object_file', type=str)
parser.add_argument('--ask_probs', action='store_true', help='Whether to ask probability per rotation.')
parser.add_argument('--save_yaml', action='store_true', help='Whether to save orientations and probabilities to yaml.')
parser.add_argument('--threshold', type=float, default=0.02, help='Threshold for including orientations or not.')
args = parser.parse_args()

mesh = trimesh.load(args.object_file)
poses = trimesh.poses.compute_stable_poses(mesh, n_samples=5, threshold=args.threshold)

urdf_file = args.object_file.replace('.obj','.urdf')
urdf_file = os.path.dirname(os.path.dirname(args.object_file)[:-1]) + '\\rigid_body.urdf'
obj = ArticulatedObject(filename=urdf_file, scale=1.0)
body_id = simulator.import_object(obj)

info_dict = {}
aabb = pb.getAABB(body_id)
print('Showing all stable placement rotations:')
dicts = []
def viz_transform(transform):
    quat = Quaternion(matrix=transform)
    r = quat.real
    v = quat.vector
    obj.set_position_orientation([0,0,mesh.extents[2]/2.0],
                                 [quat.vector[0],quat.vector[1],quat.vector[2],quat.real])
    return r,v

for i in range(len(poses[0])):
    rotation_dict = {}
    transform = poses[0][i]
    r,v = viz_transform(transform)
    if args.ask_probs:
        inp = input('Enter probability of rotation, or +/- to rotate about Z:')
        while inp[0]=='+' or inp[0]=='-':
            rot_num = float(inp[1:])
            if inp[0]=='-':
                rot_num*=-1
            z_rot = np.array([[math.cos(math.pi*rot_num), -math.sin(math.pi*rot_num), 0.0, 0.0],
                               [math.sin(math.pi*rot_num), math.cos(math.pi*rot_num), 0.0, 0.0],
                               [0.0,0.0,1.0,0.0],
                               [0.0,0.0,0.0,1.0]])
            transform = np.matmul(z_rot,transform)
            r,v = viz_transform(transform)
            inp = input('Enter probability of rotation, or +/- to rotate about Z:')
        prob = float(inp)
        rotation_dict['prob'] = prob
    else:
        skip = input('Rotation %d: (press enter to continue)'%(i+1))
    rotation_dict['rotation'] = [float(v[0]), float(v[1]), float(v[2]), float(r)]
    aabb = pb.getAABB(body_id)
    size = [aabb[1][0] - aabb[0][0] , aabb[1][1] - aabb[0][1] , aabb[1][2] - aabb[0][2]]
    print('Bounding box size=%s'%str(size))
    rotation_dict['size'] = size
    if not args.ask_probs or prob > 0:
        dicts.append(rotation_dict)

print('Summary:')
for d in dicts:
    print(d)

if args.save_yaml:
    path = os.path.dirname(args.object_file)
    output_path = os.path.join(path,'orientations.yaml')

    with open(output_path, 'w') as f:
        documents = yaml.dump(dicts, f)
