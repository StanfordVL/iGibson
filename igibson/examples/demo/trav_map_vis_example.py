import networkx as nx
import matplotlib.pyplot as plt
import cv2
import igibson
from PIL import Image
import numpy as np
import os
from igibson.utils.assets_utils import get_scene_path


def main():
    scene_id = 'Rs'
    trav_map_original_size = 1000
    trav_map_size = 200
    trav_map_erosion = 2
    floor_map = []
    floor_graph = []

    with open(os.path.join(get_scene_path(scene_id), 'floors.txt'), 'r') as f:
        floors = sorted(list(map(float, f.readlines())))
        print('floor_heights', floors)

    for f in range(len(floors)):
        trav_map = Image.open(os.path.join(get_scene_path(scene_id), 'floor_trav_{}.png'.format(f)))
        obstacle_map = Image.open(os.path.join(get_scene_path(scene_id), 'floor_{}.png'.format(f)))
        trav_map = np.array(trav_map.resize((trav_map_size, trav_map_size)))
        obstacle_map = np.array(obstacle_map.resize((trav_map_size, trav_map_size)))
        trav_map[obstacle_map == 0] = 0
        trav_map = cv2.erode(trav_map, np.ones((trav_map_erosion, trav_map_erosion)))
        plt.figure(f, figsize=(12, 12))
        plt.imshow(trav_map)

    plt.show()


if __name__ == "__main__":
    main()
