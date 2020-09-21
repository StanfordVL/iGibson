from gibson2.simulator import Simulator
from gibson2.scenes.igibson_indoor_scene import iGSDFScene
from gibson2.render.profiler import Profiler
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Name of the scene in the iG Dataset')
    args = parser.parse_args()

    s = Simulator(mode='gui', image_width=256, image_height=256, enable_shadow=True, enable_msaa=False)

    scene = iGSDFScene(args.scene)
    s.import_ig_scene(scene)

    for i in range(10000):
        with Profiler('Simulator step'):
            s.step()
    s.disconnect()


if __name__ == '__main__':
    main()
