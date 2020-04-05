import pybullet as p
from gibson2.utils.assets_utils import get_model_path, get_texture_file
import gibson2

import os
import sys
import time


def main():
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = os.path.join(get_model_path('Rs'), 'mesh_z_up.obj')

    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)

    # Load scenes
    collision_id = p.createCollisionShape(p.GEOM_MESH,
                                          fileName=model_path,
                                          meshScale=1.0,
                                          flags=p.GEOM_FORCE_CONCAVE_TRIMESH)
    visual_id = p.createVisualShape(p.GEOM_MESH,
                                    fileName=model_path,
                                    meshScale=1.0)
    texture_filename = get_texture_file(model_path)
    texture_id = p.loadTexture(texture_filename)

    mesh_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                baseVisualShapeIndex=visual_id)

    # Load robots
    turtlebot_urdf = os.path.join(gibson2.assets_path, 'models/turtlebot/turtlebot.urdf')
    robot_id = p.loadURDF(turtlebot_urdf, flags=p.URDF_USE_MATERIAL_COLORS_FROM_MTL)

    # Load objects
    obj_visual_filename = os.path.join(gibson2.assets_path, 'models/ycb/002_master_chef_can/textured_simple.obj')
    obj_collision_filename = os.path.join(gibson2.assets_path, 'models/ycb/002_master_chef_can/textured_simple_vhacd.obj')
    collision_id = p.createCollisionShape(p.GEOM_MESH,
                                          fileName=obj_collision_filename,
                                          meshScale=1.0)
    visual_id = p.createVisualShape(p.GEOM_MESH,
                                    fileName=obj_visual_filename,
                                    meshScale=1.0)
    object_id = p.createMultiBody(baseCollisionShapeIndex=collision_id,
                                  baseVisualShapeIndex=visual_id,
                                  basePosition=[1.0, 0.0, 1.0],
                                  baseMass=0.1)

    for _ in range(10000):
        p.stepSimulation()

    p.disconnect()


if __name__ == '__main__':
    main()

