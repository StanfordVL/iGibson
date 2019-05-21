from PIL import Image
import numpy as np
import OpenGL.GL as GL

colormap = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]


def loadTexture(path, scale=1):
    img = Image.open(path).transpose(Image.FLIP_TOP_BOTTOM)
    w, h = img.size

    img = img.resize((int(w * scale), int(h * scale)), Image.BICUBIC)

    img_data = np.frombuffer(img.tobytes(), np.uint8)
    #print(img_data.shape)
    width, height = img.size
    # glTexImage2D expects the first element of the image data to be the
    # bottom-left corner of the image.  Subsequent elements go left to right,
    # with subsequent lines going from bottom to top.

    # However, the image data was created with PIL Image tostring and numpy's
    # fromstring, which means we have to do a bit of reorganization. The first
    # element in the data output by tostring() will be the top-left corner of
    # the image, with following values going left-to-right and lines going
    # top-to-bottom.  So, we need to flip the vertical coordinate (y).
    texture = GL.glGenTextures(1)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR_MIPMAP_LINEAR)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_REPEAT)
    GL.glTexParameterf(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_REPEAT)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGB, width, height, 0, GL.GL_RGB,
                    GL.GL_UNSIGNED_BYTE, img_data)
    GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return texture
