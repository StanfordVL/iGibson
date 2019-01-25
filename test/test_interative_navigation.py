import yaml
from gibson2.core.physics.robot_locomotors import *
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import *
from gibson2.utils.utils import parse_config


config = parse_config('test.yaml')


def test_jr2():
    s =Simulator(mode='gui')
    scene = StadiumScene()
    s.import_scene(scene)
    jr2 = JR2_Kinova(config)
    s.import_robot(jr2)

    #for i in range(p.getNumJoints(jr2.robot_ids[0])):
    #    for j in range(p.getNumJoints(jr2.robot_ids[0])):
    #        if 'm1n6s200' in str(p.getJointInfo(jr2.robot_ids[0], i)[1]):
    #            link_id1 = p.getJointInfo(jr2.robot_ids[0], i)[0]
    #            link_id2 = p.getJointInfo(jr2.robot_ids[0], j)[0]
    #            print('set collision', i, j, link_id1, link_id2, p.getJointInfo(jr2.robot_ids[0], i))
    #            p.setCollisionFilterPair(jr2.robot_ids[0], jr2.robot_ids[0], link_id1, link_id2, 1)
    #            p.setCollisionFilterPair(jr2.robot_ids[0], jr2.robot_ids[0], link_id2, link_id1, 1)

    jr2.apply_action([0, 0, 0, 0, -1, -2, 1, 0, 0, 0, 0, 1])

    while True:
        s.step()

    s.disconnect()

test_jr2()