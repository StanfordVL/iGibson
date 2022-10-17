from igibson.scenes.igibson_indoor_scene import InteractiveIndoorScene
from igibson.simulator import Simulator
import matplotlib.pyplot as plt
import numpy as np
import cv2

IMAGE_HEIGHT = 720
IMAGE_WIDTH = 1024

sim = Simulator(
    image_height=IMAGE_HEIGHT,
    image_width=IMAGE_WIDTH,
)

scene = InteractiveIndoorScene(scene_id="Rs_int", not_load_object_categories=["door"], trav_map_type="no_door")
sim.import_scene(scene)
points_in_open_space_to_plot = []
track = 1
trav_map = sim.scene.floor_map[0]
print(np.array(sim.scene.floor_map).shape)
while track <= 5000:
    sim.step()
    if track%100 == 0:
        camera_position = sim.scene.world_to_map((sim.viewer.px, sim.viewer.py))
        #breakpoint()
        points_in_open_space_to_plot.append(camera_position)
        # print(sim.viewer.renderer.camera)
        img = cv2.cvtColor(trav_map, cv2.COLOR_GRAY2RGB)
        cv2.circle(img, (camera_position[1], camera_position[0]), 3, color=(255, 0, 0), thickness=-1)
        cv2.imshow("Test", img)
        cv2.waitKey(1)
        track += 1
    track += 1
plt.imshow(trav_map)
points_in_open_space_to_plot = np.array(points_in_open_space_to_plot)
plt.scatter(points_in_open_space_to_plot[:,0], points_in_open_space_to_plot[:, 1], color="#62b41f")
plt.show()

