from gibson2.robots.turtlebot_robot import Turtlebot
from gibson2.simulator import Simulator
from gibson2.scenes.gibson_indoor_scene import StaticIndoorScene
from gibson2.scenes.igibson_indoor_scene import InteractiveIndoorScene
from gibson2.objects.ycb_object import YCBObject
from gibson2.objects.soft_object import SoftObject
from gibson2.utils.utils import parse_config
from gibson2.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
import numpy as np
from gibson2.render.profiler import Profiler
import gibson2
import os
import pybullet as p
import pybullet_data


def main():
    config = parse_config(os.path.join(gibson2.example_config_path, 'turtlebot_demo.yaml'))

    settings = MeshRendererSettings(enable_shadow=True, msaa=False, optimized=False)
    s = Simulator(mode='gui', image_width=512,
                  image_height=512, rendering_settings=settings)

    p.resetSimulation(p.RESET_USE_DEFORMABLE_WORLD)
    p.setGravity(0,0,-10)
    p.setTimeStep(0.001)
    p.setPhysicsEngineParameter(sparseSdfVoxelSize=0.05)

    p.setAdditionalSearchPath(pybullet_data.getDataPath())


    scene = InteractiveIndoorScene(
        'Rs_int', texture_randomization=False, object_randomization=False)

    scene._set_first_n_objects(3)
    s.import_ig_scene(scene)

    #planeOrn = [0, 0, 0, 1]  # p.getQuaternionFromEuler([0.3,0,0])
    #planeId = p.loadURDF("plane.urdf", [0, 0, 0], planeOrn)

    tmp = YCBObject
    obj = SoftObject(fileName="Provence_Bath_Towel_Royal_Blue_cm.obj", simFileName = "Provence_Bath_Towel_Royal_Blue_cm.obj", basePosition=[0, 0, 1.5],
                     mass=1, scale=0.1, collisionMargin=0.04, useNeoHookean=0, useBendingSprings=1, useMassSpring=1, useFaceContact=1,
                     useSelfCollision=1, springElasticStiffness=40, springDampingStiffness=0.1, springDampingAllDirections=0,frictionCoeff=1.0)
                     
    #obj = SoftObject(fileName="ball.obj", simFileName = "ball_processed.obj", basePosition=[0, 0, 1.5],
    #                 scale=0.2, mass=0.5, useNeoHookean=1,NeoHookeanMu=400, NeoHookeanLambda=600,
    #                 NeoHookeanDamping=0.001, useSelfCollision=1,
    #                 frictionCoeff=.5, collisionMargin=0.001)
    s.import_object(obj)
    for _ in range(3):
        obj = YCBObject('003_cracker_box')
        s.import_object(obj)
        obj.set_position_orientation(np.random.uniform(
            low=0, high=2, size=3), [0, 0, 0, 1])


    while True:
        s.step()
        p.setGravity(0, 0, -10)


    s.disconnect()


if __name__ == '__main__':
    main()
