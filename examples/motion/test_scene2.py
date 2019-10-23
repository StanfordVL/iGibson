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
    hyperion_path = '/data4/hyperion'
    files = [item for item in os.listdir(hyperion_path) if not 'vhacd' in item and 'centered' in item and not 'py' in item]
    vhacd_files = [item.split('.')[0] + '_vhacd.obj' for item in files]

    p.setTimeStep(0.05)

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

    s.viewer.renderer.material_string = '''
material orangish
{
    color 1.0 0.186 0.0
    roughness 0.035
    specular 0.5
    clearcoat 1.0
    clearcoatGloss 0.93
}

material glass
{
    color 1.0 1.0 1.0
    brdf 1
}

material silver
{
    color 0.9 0.9 0.9
    specular 0.5
    roughness 0.01
    metallic 1.0 
}

material ring_silver
{
    color 1.0 1.0 1.0
    roughness 0.01 
    specular 0.5
    metallic 1.0 
}

material cream
{
    color 1.0 0.94 0.8
    roughness 1.0
    specular 0.5
}

material ping
{
    #color 1.0 1.0 1.0
    #roughness 0.8
    #subsurface 1.0
    #specular 0.5
    
    color 0.93 0.89 0.85
    specular 0.6
    roughness 0.2
    subsurface 0.4
}

material marb1
{
    color 0.026 0.147 0.075
    roughness 0.077
    specular 0.5
    subsurface 1.0
    clearcoat 1.0
    clearcoatGloss 0.93
}

material marb2
{
    color 0.099 0.24 0.134
    roughness 0.077
    specular 0.5
    subsurface 1.0
    clearcoat 1.0
    clearcoatGloss 0.93
}
'''
    
    s.viewer.renderer.light_string = '''
light
{
    emission 80.0 80.0 80.0
    position 0 -439.0 390.0
    radius 60.0
    type Sphere
}
'''

    s.viewer.renderer.mmap = dict(zip(['/data4/hyperion/marb2_centered.obj',
    '/data4/hyperion/marb1_centered.obj',
    '/data4/hyperion/ring1_centered.obj',
    '/data4/hyperion/pingpong_centered.obj',
    '/data4/hyperion/glass_centered.obj',
    '/data4/hyperion/chrome_centered.obj',
    '/data4/hyperion/dragon_centered.obj',
    '/data4/hyperion/orange_centered.obj',
    '/data4/hyperion/ring2_centered.obj',
    '/data4/hyperion/ring3_centered.obj',
    '/data4/hyperion/plate_centered.obj']
    ,['marb2', 'marb1', 'ring_silver', 'ping', 'glass', 'silver', 'glass', 'orangish', 'ring_silver', 'ring_silver', 'cream']))

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

        s.step()
        

        for step in range(200):
            s.step()

            instances = [instance.objects[0].filename for instance in s.viewer.renderer.instances]
            poses_trans = [instance.poses_trans[0] for instance in s.viewer.renderer.instances]
            poses_rot = [instance.poses_rot[0] for instance in s.viewer.renderer.instances]
            if step % 2 == 0:
                s.viewer.renderer.save_pose()
            
            print(instances)
            print(poses_trans)
            print(poses_rot)
            s.viewer.renderer.export_scene('test_{:04d}.scene'.format(step))
            
            if step % 2 == 1:
                s.viewer.renderer.load_pose()
                Image.fromarray((255*s.viewer.renderer.render(modes=('rgb'))[0][:,:,:3]).astype(np.uint8)).save('test_rgb_{:04d}.png'.format(step))
                Image.fromarray((255*s.viewer.renderer.render(modes=('normal'))[0][:,:,:3]).astype(np.uint8)).save('test_normal_{:04d}.png'.format(step))

            
        #Image.fromarray((255*s.viewer.renderer.render(modes=('rgb'))[0][:,:,:3]).astype(np.uint8)).save('restored.png')
        #Image.fromarray((255*s.viewer.renderer.render(modes=('normal'))[0][:,:,:3]).astype(np.uint8)).save('restored_normal.png'.format(step))

    finally:
        s.disconnect()
