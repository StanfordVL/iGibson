import numpy as np

from igibson.render.mesh_renderer.mesh_renderer_settings import MeshRendererSettings
from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator


def main():
    settings = MeshRendererSettings(optimized=True)
    s = Simulator(mode="gui", image_width=512, image_height=512, device_idx=0, rendering_settings=settings)
    scene = InteractiveIndoorScene("Rs_int", texture_randomization=False, object_randomization=False)
    s.import_ig_scene(scene)
    np.random.seed(0)
    for _ in range(10):
        pt = scene.get_random_point_by_room_type("living_room")[1]
        print("random point in living_room", pt)

    for i in range(1000):
        s.step()
        if i % 100 == 0:
            for obj in scene.objects_by_category["window"]:
                obj.highlight()

        if i % 100 == 50:
            for obj in scene.objects_by_category["window"]:
                obj.unhighlight()

    s.disconnect()


if __name__ == "__main__":
    main()
