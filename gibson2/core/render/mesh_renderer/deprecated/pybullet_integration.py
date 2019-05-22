import pybullet as p
import sys
import time
import numpy as np
from gibson2.core.render.mesh_renderer.mesh_renderer_cpu import *
import cv2

p.connect(p.GUI)

filename = sys.argv[1]
collisionId = p.createCollisionShape(p.GEOM_MESH,
                                     fileName=filename,
                                     meshScale=1,
                                     flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
boundaryUid = p.createMultiBody(baseCollisionShapeIndex=collisionId, baseVisualShapeIndex=-1)
p.changeDynamics(boundaryUid, -1, lateralFriction=1)

object_filename = '/home/fei/Downloads/models/011_banana/textured_simple2.obj'

collision_obj_list = []
collisionId2 = p.createCollisionShape(p.GEOM_MESH, fileName=object_filename, meshScale=1)

for i in range(20):
    x = np.random.uniform()
    y = np.random.uniform()
    z = 0.5 + 2 * np.random.uniform()
    boundaryUid2 = p.createMultiBody(basePosition=[x, y, z],
                                     baseMass=0.1,
                                     baseCollisionShapeIndex=collisionId2,
                                     baseVisualShapeIndex=-1)
    collision_obj_list.append(boundaryUid2)

model_path = sys.argv[1]
renderer = MeshRenderer(width=800, height=600)
renderer.load_object(model_path)

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
            r2 = np.array([[np.cos(-dx), -np.sin(-dx), 0], [np.sin(-dx),
                                                            np.cos(-dx), 0], [0, 0, 1]])
            view_direction = r1.dot(r2).dot(view_direction)
    elif event == cv2.EVENT_LBUTTONUP:
        down = False


cv2.namedWindow('test')
cv2.setMouseCallback('test', change_dir)

p.setTimeStep(1 / 240.0)

renderer_obj_ids = []

for i in range(20):
    x = np.random.uniform()
    y = np.random.uniform()
    z = 0.5 + 2 * np.random.uniform()
    model_path2 = '/home/fei/Downloads/models/011_banana/textured_simple.obj'
    object_ids = renderer.load_object(model_path2, scale=1)
    print(object_ids)
    renderer_obj_ids.append(object_ids[0])

while True:
    p.setGravity(0, 0, -10)
    p.stepSimulation()

    for i, ids in enumerate(collision_obj_list):
        pos, orn = p.getBasePositionAndOrientation(ids)
        renderer.set_pose([pos[0], pos[1], pos[2], orn[-1], orn[0], orn[1], orn[2]],
                          renderer_obj_ids[i])    #[x y z quat]

    frame = renderer.render()
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
