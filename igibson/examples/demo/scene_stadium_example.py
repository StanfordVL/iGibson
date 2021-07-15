from igibson.scenes.stadium_scene import StadiumScene
import pybullet as p
import time


def main():
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    p.setTimeStep(1.0 / 240.0)

    scene = StadiumScene()
    scene.load()

    for _ in range(24000):  # at least 100 seconds
        p.stepSimulation()
        time.sleep(1.0 / 240.0)

    p.disconnect()


if __name__ == "__main__":
    main()
