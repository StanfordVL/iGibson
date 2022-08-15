import pickle as pkl
import cv2
import numpy as np
def load_av_map(path):
    map_data = pkl.load(open(path, 'rb'))
    print(map_data['semantic_map'].dtype)
    print(map_data['top_down_map'].shape)
    cv2.imwrite("semantic.png", (255/13*map_data['semantic_map']).astype(np.uint8))
    cv2.imwrite("top_down.png", (255*map_data['top_down_map']).astype(np.uint8))


if __name__ == "__main__":
    load_av_map("topdownmap/17DRP5sb8fy_occant_semantic.pkl")
