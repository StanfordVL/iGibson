from gibson2.simulator import Simulator
from gibson2.scenes.scene_base import GSDFScene
from gibson2.utils.utils import parse_config
from gibson2.render.profiler import Profiler
import argparse



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='scene urdf file')
    args = parser.parse_args()

    config = parse_config('../configs/turtlebot_demo.yaml')
    s = Simulator(mode='gui', image_width=256, image_height=256, enable_shadow=True, enable_msaa=False)

    scene = GSDFScene(scene_file=args.scene)
    s.import_scene(scene)

    for i in range(10000):
        with Profiler('Simulator step'):
            s.step()
    s.disconnect()


if __name__ == '__main__':
    main()
