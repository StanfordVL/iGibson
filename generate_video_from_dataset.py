import cv2
import h5py
import numpy as np
from PIL import Image

filename = "data/Pomaria_0_int/0/data.hdf5"
frame_size = (1024, 720)
fourcc = cv2.VideoWriter_fourcc(*"MP4V")
output_video = cv2.VideoWriter("output_video.mp4", fourcc, 20.0, frame_size)

with h5py.File(filename, "r") as dataset:
    for img in dataset["rgb"]:
        output_video.write(np.uint8(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
    output_video.release()
