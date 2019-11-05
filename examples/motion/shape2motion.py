from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.core.physics.interactive_objects import BoxShape, YCBObject, RBOObject, InteractiveObj, Pedestrian
from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova
import yaml
import gibson2
import os


if __name__ == "__main__":
    s = Simulator(mode='gui', resolution=512)
    try:
        scene = StadiumScene()
        s.import_scene(scene)

        for i in range(1,31):
            fn = "/data4/mdv0/cabinet/{:04d}/part_objs/cabinet_{:04d}.urdf".format(i,i)
            print(fn)
            obj = InteractiveObj(filename=fn)
            s.import_interactive_object(obj)
            obj.set_position([2*i//6, 2*(i%6), 2])

        while True:
            s.step()
    finally:
        s.disconnect()