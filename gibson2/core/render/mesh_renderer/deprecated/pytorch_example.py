import sys
import ctypes
from contextlib import contextmanager
from PIL import Image
import mesh_renderer.glutils.glcontext as glcontext
import pycuda.driver
from pycuda.gl import graphics_map_flags
import torch
import OpenGL.GL as GL
import cv2
import numpy as np
from pyassimp import *
from gibson2.core.render.mesh_renderer.glutils.meshutil import perspective, lookat, xyz2mat, quat2rotmat, mat2xyz, safemat2quat
from transforms3d.quaternions import axangle2quat, mat2quat
from transforms3d.euler import quat2euler, mat2euler
from gibson2.core.render.mesh_renderer import CppMeshRenderer
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
from gibson2.core.render.mesh_renderer.get_available_devices import get_available_devices, get_cuda_device
from gibson2.core.render.mesh_renderer_tensor import *
import torch.nn as nn
MAX_NUM_OBJECTS = 3


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(4, 4, stride=1, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv1(x)


if __name__ == '__main__':
    model_path = sys.argv[1]
    w = 400
    h = 300

    renderer = MeshTensorRenderer(w, h)
    renderer.load_object(model_path)

    camera_pose = np.array([0, 0, 1.2])
    view_direction = np.array([1, 0, 0])
    renderer.set_camera(camera_pose, camera_pose + view_direction, [0, 0, 1])
    renderer.set_fov(90)
    tensor = torch.cuda.ByteTensor(h, w, 4)
    tensor2 = torch.cuda.ByteTensor(h, w, 4)

    pytorch_model = Net()
    pytorch_model.cuda(tensor.device)
    torch.cuda.synchronize()
    print(pytorch_model)

    start = time.time()
    for _ in tqdm(range(3000)):
        renderer.render(tensor, tensor2)
        torch.cuda.synchronize()
        res = pytorch_model.forward(tensor.float().permute(2, 0, 1).unsqueeze(0))
        #print(res.size())
        torch.cuda.synchronize()

    dt = time.time() - start
    print("{} fps".format(3000 / dt))

    img_np = tensor.flip(0).data.cpu().numpy().reshape(h, w, 4)
    img_np2 = tensor2.flip(0).data.cpu().numpy().reshape(h, w, 4)
    print(img_np.shape)
    plt.imshow(np.concatenate([img_np, img_np2], axis=1))
    plt.show()
    renderer.release()
