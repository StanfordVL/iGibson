import cv2
import numpy as np

im = cv2.imread("result/images/savi_2.png")

new_im = np.zeros_like(im)
new_im[:,:,0] = im[:,:,2]
new_im[:,:,1] = im[:,:,1]
new_im[:,:,2] = im[:,:,0]
print(new_im[:,:,2].min(), new_im[:,:,2].min())
# new_im[np.all(new_im == (27, 55, 229), axis=-1)] = (233,55,31)

cv2.imwrite("result/images/filped.png", new_im)