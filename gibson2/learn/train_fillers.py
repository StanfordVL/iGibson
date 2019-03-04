import argparse
import os
import re
import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torchvision.utils as vutils
from gibson2.data.datasets import RenderPairDataset
from gibson2.learn.completion import CompletionNet, identity_init, Perceptual
from tensorboardX import SummaryWriter
from datetime import datetime
import gibson2.learn.vision_utils
import torch.nn.functional as F
import torchvision.models as models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', required=True, help='path to dataset')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--imgsize', type=int, default=256, help='image size')
    parser.add_argument('--nf', type=int, default=64, help='number of filters')
    parser.add_argument('--batchsize', type=int, default=20, help='batchsize')
    parser.add_argument('--workers', type=int, default=9, help='number of workers')
    parser.add_argument('--nepoch', type=int, default=50, help='number of epochs')
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate, default=0.002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', type=str, default="filler_pano_pc_full", help='output folder')
    parser.add_argument('--model', type=str, default="", help='model path')
    parser.add_argument('--cepoch', type=int, default=0, help='current epoch')
    parser.add_argument('--loss', type=str, default="perceptual", help='l1 only')
    parser.add_argument('--init', type=str, default="iden", help='init method')
    parser.add_argument('--l1', type=float, default=0, help='add l1 loss')
    parser.add_argument('--color_coeff', type=float, default=0, help='add color match loss')
    parser.add_argument('--unfiller', action='store_true', help='debug mode')
    parser.add_argument('--joint', action='store_true', help='debug mode')
    parser.add_argument('--use_depth', action='store_true', default=False, help='debug mode')
    parser.add_argument('--zoom', type=int, default=1, help='debug mode')
    parser.add_argument('--patchsize', type=int, default=256, help='debug mode')

    mean = torch.from_numpy(np.array([0.57441127, 0.54226291, 0.50356019]).astype(np.float32)).clone()
    opt = parser.parse_args()
    print(opt)
    writer = SummaryWriter(opt.outf + '/runs/' + datetime.now().strftime('%B%d  %H:%M:%S'))
    try:
        os.makedirs(opt.outf)
    except OSError:
        pass

    zoom = opt.zoom
    patchsize = opt.patchsize

    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
    ])

    d = RenderPairDataset(root=opt.dataroot, transform=tf)
    d_test = RenderPairDataset(root=opt.dataroot, transform=tf, train=False)

    cudnn.benchmark = True

    dataloader = torch.utils.data.DataLoader(d, batch_size=opt.batchsize, shuffle=True, num_workers=int(opt.workers),
                                             drop_last=True, pin_memory=False)
    dataloader_test = torch.utils.data.DataLoader(d_test, batch_size=opt.batchsize, shuffle=True,
                                                  num_workers=int(opt.workers), drop_last=True, pin_memory=False)

    comp = CompletionNet(norm=nn.BatchNorm2d, nf=opt.nf)

    current_epoch = opt.cepoch
    comp = torch.nn.DataParallel(comp).cuda()
    comp.apply(weights_init)

    if opt.model != '':
        comp.load_state_dict(torch.load(opt.model))
        # dis.load_state_dict(torch.load(opt.model.replace("G", "D")))
        current_epoch = opt.cepoch

    l2 = nn.MSELoss()
    # if opt.loss == 'train_init':
    #    params = list(comp.parameters())
    #    sel = np.random.choice(len(params), len(params)/2, replace=False)
    #    params_sel = [params[i] for i in sel]
    #    optimizerG = torch.optim.Adam(params_sel, lr = opt.lr, betas = (opt.beta1, 0.999))
    #
    # else:
    optimizerG = torch.optim.Adam(comp.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    curriculum = (200000, 300000)  # step to start D training and G training, slightly different from the paper
    alpha = 0.004

    vgg16 = models.vgg16(pretrained=False)
    vgg16.load_state_dict(torch.load('vgg16-397923af.pth'))
    feat = vgg16.features
    p = torch.nn.DataParallel(Perceptual(feat)).cuda()

    for param in p.parameters():
        param.requires_grad = False

    imgnet_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406]).astype(np.float32)).clone()
    imgnet_std = torch.from_numpy(np.array([0.229, 0.224, 0.225]).astype(np.float32)).clone()

    imgnet_mean_img = Variable(imgnet_mean.view(1, 3, 1, 1).repeat(opt.batchsize * 4, 1, patchsize, patchsize)).cuda()
    imgnet_std_img = Variable(imgnet_std.view(1, 3, 1, 1).repeat(opt.batchsize * 4, 1, patchsize, patchsize)).cuda()
    test_loader_enum = enumerate(dataloader_test)

    for epoch in range(current_epoch, opt.nepoch):
        for i, data in enumerate(dataloader, 0):
            optimizerG.zero_grad()
            source = data[0].cuda()
            target = data[1].cuda()
            step = i + epoch * len(dataloader)

            recon = comp(source)

            loss = l2(p(recon), p(target).detach()) + opt.l1 * l2(recon, target)
            loss.backward(retain_graph=True)
            optimizerG.step()

            print('[%d/%d][%d/%d] %d MSEloss: %f' % (epoch, opt.nepoch, i, len(dataloader), step, loss.data.item()))

            if i % 10 == 0:
                writer.add_scalar('MSEloss', loss.data.item(), step)

            if i % 100 == 0:
                visual = torch.cat([source.cpu().data, target.cpu().data, recon.cpu().data], 3)
                visual = vutils.make_grid(visual)
                vutils.save_image(visual, '%s/compare%d_%d.png' % (opt.outf, epoch, i), nrow=1)

            if i % 2000 == 0:
                torch.save(comp.state_dict(), '%s/compG_epoch%d_%d.pth' % (opt.outf, epoch, i))


if __name__ == '__main__':
    main()
