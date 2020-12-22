from gibson2.core.physics.scene import BuildingScene
import pybullet as p
import numpy as np
import time

def main():
    scenes = ['Bolton', 'Connellsville', 'Pleasant', 'Cantwell', 'Placida', 'Nicut', 'Brentsville', 'Samuels', 'Oyens', 'Kerrtown']
    for scene in scenes:
        print('scene: ', scene, '-' * 50)
        p.connect(p.DIRECT)
        p.setGravity(0,0,-9.8)
        p.setTimeStep(1./240.)

        scene = BuildingScene(scene,
                              is_interactive=True,
                              build_graph=True,
                              pybullet_load_texture=True)
        scene.load()
        p.disconnect()

if __name__ == '__main__':
    main()
