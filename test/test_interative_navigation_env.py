import yaml
from gibson2.core.physics.robot_locomotors import Turtlebot, Husky, Ant, Humanoid, JR2, JR2_Kinova
from gibson2.core.simulator import Simulator
from gibson2.core.physics.scene import BuildingScene, StadiumScene
from gibson2.utils.utils import parse_config
from gibson2.core.physics.interactive_objects import InteractiveObj
from gibson2.envs.locomotor_env import NavigateEnv, NavigateRandomEnv, InteractiveNavigateEnv
import os
import gibson2
import pybullet as p

config = parse_config('test.yaml')


def test_jr2():
    config_filename = os.path.join(os.path.dirname(gibson2.__file__),
                                   '../examples/configs/jr_interactive_nav.yaml')
    nav_env = InteractiveNavigateEnv(config_file=config_filename,
                                     mode='gui',
                                     action_timestep=1.0 / 10.0,
                                     physics_timestep=1 / 40.0)
    

    #from IPython import embed; embed()


    try:
        nav_env.reset()
        for i in range(10000):    # 300 steps, 30s world time
            action = nav_env.action_space.sample()
            state, reward, done, _ = nav_env.step(action)

            #for item in p.getContactPoints():
            #    flag, bA, bB, lA, lB = item[:5]
            #    if bA == bB:
            #        print(bA, bB, lA, lB)
            #        print(p.getJointInfo(bA, lA)[1], p.getJointInfo(bA, lB)[1])

            #if done:
            #    print('Episode finished after {} timesteps'.format(i + 1))
    finally:
        nav_env.clean()


"""
    s =Simulator(mode='gui')
    scene = BuildingScene('Ohoopee')
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


    obj3 = InteractiveObj(os.path.join(gibson2.assets_path, 'models', 'scene_components', 'door.urdf'), scale=2)
    s.import_interactive_object(obj3)
    obj3.set_position_rotation([-5, -1, 0], [0, 0, np.sqrt(0.5), np.sqrt(0.5)])

    jr2.set_position([-6,0,0.1])
    #from IPython import embed; embed()


    jr2.apply_action([0.005, 0.005, 0, 0, 0,0,0, 0, 0, 0, 0, 0])
    for _ in range(400):
        s.step()

    jr2.apply_action([0,0, 0,0, 3.3607160552645428, 3.3310046132823998, 3.1408197119196117, -1.37402907967774, -0.8377005721485424, -1.9804208517373096, 0.09322135043256494, 2.62937740156038])

    for _ in range(400):
        s.step()

    s.disconnect()

    """

if __name__ == "__main__":
    test_jr2()
