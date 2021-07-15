from igibson.objects.ycb_object import YCBObject
from igibson.objects.articulated_object import ArticulatedObject
import igibson
import os
import pybullet as p
import pybullet_data
import time


def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    floor = os.path.join(pybullet_data.getDataPath(), "mjcf/ground_plane.xml")
    p.loadMJCF(floor)

    cabinet_0007 = os.path.join(
        igibson.assets_path, 'models/cabinet2/cabinet_0007.urdf')
    cabinet_0004 = os.path.join(
        igibson.assets_path, 'models/cabinet/cabinet_0004.urdf')

    obj1 = ArticulatedObject(filename=cabinet_0007)
    obj1.load()
    obj1.set_position([0, 0, 0.5])

    obj2 = ArticulatedObject(filename=cabinet_0004)
    obj2.load()
    obj2.set_position([0, 0, 2])

    obj3 = YCBObject('003_cracker_box')
    obj3.load()
    obj3.set_position_orientation([0, 0, 1.2], [0, 0, 0, 1])

    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


if __name__ == '__main__':
    main()
