from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.core.physics.interactive_objects import BoxShape, YCBObject, RBOObject, InteractiveObj, Pedestrian, GeneralObject
from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova
import yaml
import gibson2
import os
import pybullet as p
import pybullet_data
import numpy as np
import time
from PIL import Image


if __name__ == "__main__":
    s = Simulator(mode='gui', resolution=1024)
    hyperion_path = '/home/fei/Development/optix/Optix-PathTracer/src/data/hyperion'
    files = [item for item in os.listdir(hyperion_path) if not 'vhacd' in item and 'centered' in item and not 'py' in item]
    vhacd_files = [item.split('.')[0] + '_vhacd.obj' for item in files]

    planeName = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    ground_plane_mjcf = p.loadMJCF(planeName)
    print(files)
    print(ground_plane_mjcf)
    pos_files = [item.split('_')[0] + '_pos.txt' for item in files]
    positions = [np.loadtxt(os.path.join(hyperion_path, item)) for item in pos_files]

    camera_pose = np.array([-12.4616, 15.8139, 18.4849])
    view_direction = np.array([ -8.90342e-07, 1.53424, 1.13471e-06]) - camera_pose
    s.viewer.renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    s.viewer.renderer.set_fov(35)
    s.viewer.free_view_point = False
    try:
        for i in range(len(files)):
            
            if  'floor' in files[i]:
                continue

            obj =  GeneralObject(os.path.join(hyperion_path, files[i]), os.path.join(hyperion_path, vhacd_files[i]))
            body_id = s.import_interactive_object(obj)
            offset = 1

            if not 'floor' in files[i] and not 'plate' in files[i]:
                offset += 3
            else:
                p.changeDynamics(body_id, -1, mass=100)

            p.changeDynamics(body_id, -1, restitution=0.95)
            p.resetBasePositionAndOrientation(body_id, [positions[i][2],positions[i][0],positions[i][1]+offset], [0.5, 0.5, 0.5, 0.5])


            #if 'floor' in files[i]:
            #    p.createConstraint(ground_plane_mjcf[0],-1, body_id, -1, p.JOINT_FIXED, [0,0,1], [0,0,0], [0,0,0])
            #if 'plate' in files[i]:
            #    p.createConstraint(ground_plane_mjcf[0],-1, body_id, -1, p.JOINT_FIXED, [0,0,1], [0,0,0], [0,0,0])
            
        #while True:
        s.step()
        
        instances = [instance.objects[0].filename for instance in s.viewer.renderer.instances]
        poses_trans = [instance.poses_trans[0] for instance in s.viewer.renderer.instances]
        poses_rot = [instance.poses_rot[0] for instance in s.viewer.renderer.instances]

        print(instances)
        print(poses_trans)
        print(poses_rot)
        s.viewer.renderer.export_scene('test.scene')
        
        Image.fromarray((255*s.viewer.renderer.render(modes=('rgb'))[0][:,:,:3]).astype(np.uint8)).save('test_rgb.png')
        Image.fromarray((255*s.viewer.renderer.render(modes=('normal'))[0][:,:,:3]).astype(np.uint8)).save('test_normal.png')
    

    finally:
        s.disconnect()