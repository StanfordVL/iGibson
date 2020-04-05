from gibson2.core.physics.interactive_objects import InteractiveObj, YCBObject
import gibson2
import os
import pybullet as p
import time

def main():
    p.connect(p.GUI)
    p.setGravity(0,0,-9.8)
    p.setTimeStep(1./240.)

    cabinet_0007 = os.path.join(gibson2.assets_path, 'models/cabinet2/cabinet_0007.urdf')
    cabinet_0004 = os.path.join(gibson2.assets_path, 'models/cabinet/cabinet_0004.urdf')

    obj1 = InteractiveObj(filename=cabinet_0007)
    obj1.load()
    obj1.set_position([0,0,0.5])

    obj2 = InteractiveObj(filename=cabinet_0004)
    obj2.load()
    obj2.set_position([0,0,2])

    obj3 = YCBObject('003_cracker_box')
    obj3.load()
    p.resetBasePositionAndOrientation(obj3.body_id, [0,0,1.2], [0,0,0,1])

    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


if __name__ == '__main__':
    main()
