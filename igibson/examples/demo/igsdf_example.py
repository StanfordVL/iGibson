from igibson.simulator import Simulator
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.render.profiler import Profiler
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene', type=str, help='Name of the scene in the iG Dataset')
    args = parser.parse_args()
    settings = MeshRendererSettings(enable_shadow=True, msaa=False)
    s = Simulator(mode='gui', image_width=256, image_height=256, rendering_settings=settings)

    scene = InteractiveIndoorScene(args.scene)
    s.import_ig_scene(scene)

    for i in range(10000):
        with Profiler('Simulator step'):
            s.step()
    s.disconnect()


if __name__ == '__main__':
    main()
