import torch
import torch.nn as nn
import numpy as np
from gibson2.learn.model import UNet
import gibson2
import os
model = UNet(input_channels=3)
model = torch.nn.DataParallel(model).cuda()
checkpoint = torch.load(os.path.join(gibson2.assets_path, 'networks', 'ckpt_0008.pth.tar'))
model.load_state_dict(checkpoint['state_dict'])
model.eval()
softmax = nn.Softmax(dim=1).cuda()
with torch.no_grad():
    images = torch.from_numpy(
                np.random.random([5,3,128,128]).astype(np.float32)).cuda()
    pred, features = model(images)
    # prediction is the dense prediction (pre softmax)
    # (5,2,128,128)
    # features are the intermediate features
    # (5,32,128,128)
    pred_softmaxed = softmax(pred)
    # prediction softmaxed
    # (5,2,128,128)

