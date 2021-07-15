from igibson.scenes.gibson_indoor_scene import StaticIndoorScene
import pybullet as p
import numpy as np
import time


def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1./240.)

    scene = StaticIndoorScene('Rs',
                              build_graph=True,
                              pybullet_load_texture=True)
    scene.load()

    np.random.seed(0)
    for _ in range(10):
        random_floor = scene.get_random_floor()
        p1 = scene.get_random_point(random_floor)[1]
        p2 = scene.get_random_point(random_floor)[1]
        shortest_path, geodesic_distance = scene.get_shortest_path(
            random_floor, p1[:2], p2[:2], entire_path=True)
        print('random point 1:', p1)
        print('random point 2:', p2)
        print('geodesic distance between p1 and p2', geodesic_distance)
        print('shortest path from p1 to p2:', shortest_path)

    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1./240.)

    p.disconnect()


if __name__ == '__main__':
    main()
