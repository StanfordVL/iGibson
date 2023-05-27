import cv2
import numpy as np

def main():
    vidcap = cv2.VideoCapture('occ_gt_video.mp4')
    success,image = vidcap.read()
    image = np.round(image / 255.0).astype(np.uint8)
    print(image[:,:,0])
    # print(image[:,:,1].min())
    # print(image[:,:,2].min())
    image[np.all(image == (0, 0, 0), axis=-1)] = (139,0,0)
    # image[np.all(image == (0, 255, 0), axis=-1)] = (25,230,25)
    image[np.all(image == (1, 1, 1), axis=-1)] = (25,230,25)
    count = 0
    cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file      


if __name__ == "__main__":
    main()