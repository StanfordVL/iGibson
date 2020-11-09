import json
from IPython import embed
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import random
import time
from torch.utils.tensorboard import SummaryWriter
import shutil

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--root-dir',
                    help='root dir to store all experiments', default='bc_results')
parser.add_argument('--exp-dir',
                    help='experiment directory', required=True)
parser.add_argument('--data', metavar='DIR',
                    help='path to dataset', default='bc_data')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                    'batch size of all GPUs on the current node when '
                    'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--reg-loss-weight', default=1.0, type=float,
                    help='weight for regression loss for arm action')
parser.add_argument('--cls-loss-weight', default=1.0, type=float,
                    help='weight for classification loss for gripper action')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_f1 = 0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix="", writer=None):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.writer = writer

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def log_tensorboard(self, epoch):
        if self.writer is not None:
            for meter in self.meters:
                self.writer.add_scalar(meter.name, meter.avg, epoch)

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def get_cls_metrics(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        accu = pred.eq(target)
        target_true = torch.sum(target).float()
        pred_true = torch.sum(pred).float()
        correct_true = torch.sum(accu * pred.byte()).float()
        precision = correct_true / pred_true
        recall = correct_true / target_true
        f1_score = (2 * precision * recall) / (precision + recall)
        accu = accu.float().mean()
        return accu, precision, recall, f1_score


class BCDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir, dataset_type, input_mean=None, input_std=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.dataset_type = dataset_type
        with open('{}/{}.json'.format(root_dir, dataset_type), 'r') as f:
            self.data = json.load(f)
            for key in self.data:
                if key == 'gripper_action':
                    self.data[key] = np.array(self.data[key]).astype(np.int64)
                else:
                    self.data[key] = np.array(
                        self.data[key]).astype(np.float32)

        if dataset_type == 'train':
            input_mean = np.mean(self.data['obj_pos'], axis=0)
            input_std = np.std(self.data['obj_pos'], axis=0)
        self.input_mean = input_mean
        self.input_std = input_std

    def __len__(self):
        return len(self.data['obj_pos'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        obj_pos = self.data['obj_pos'][idx]
        arm_action = self.data['arm_action'][idx]
        gripper_action = self.data['gripper_action'][idx]
        obj_pos = (obj_pos - self.input_mean) / self.input_std

        return obj_pos, arm_action, gripper_action


class Model(nn.Module):
    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers,
                 arm_action_size,
                 gripper_action_size):
        super(Model, self).__init__()
        assert num_layers > 0
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        for i in range(1, num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.arm_action_head = nn.Linear(hidden_size, arm_action_size)
        self.gripper_action_head = nn.Linear(hidden_size, gripper_action_size)

    def forward(self, x):
        x = self.layers(x)
        arm_action = self.arm_action_head(x)
        gripper_action = self.gripper_action_head(x)
        return arm_action, gripper_action


def save_checkpoint(state, ckpt_dir, filename, is_best):
    torch.save(state, os.path.join(ckpt_dir, filename))
    if is_best:
        shutil.copyfile(os.path.join(ckpt_dir, filename),
                        os.path.join(ckpt_dir, 'model_best.pth.tar'))


def main():
    global best_f1

    args = parser.parse_args()
    root_dir = args.root_dir
    exp_dir = os.path.join(root_dir, args.exp_dir)
    ckpt_dir = os.path.join(exp_dir, 'ckpt')
    summary_dir = os.path.join(exp_dir, 'summary')
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(summary_dir, exist_ok=True)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True

    model = Model(input_size=3,
                  hidden_size=64,
                  num_layers=2,
                  arm_action_size=3,
                  gripper_action_size=2)

    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu)
    else:
        model = torch.nn.DataParallel(model).cuda()

    reg_criterion = nn.MSELoss().cuda(args.gpu)
    # cls_criterion = nn.CrossEntropyLoss(
    #     torch.tensor([0.033050047214353166, 0.9669499527856469])).cuda(args.gpu)
    cls_criterion = nn.CrossEntropyLoss(
        torch.tensor([0.3, 0.7])).cuda(args.gpu)
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            best_f1 = checkpoint['best_f1']
            if args.gpu is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_f1 = best_f1.to(args.gpu)
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    train_dataset = BCDataset(args.data, 'train')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_dataset = BCDataset(
        args.data, 'val', train_dataset.input_mean, train_dataset.input_std)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, reg_criterion, cls_criterion, -1, args)
        return

    writer = SummaryWriter(summary_dir)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, reg_criterion,
              cls_criterion, optimizer, epoch, args, writer)

        # evaluate on validation set
        f1 = validate(val_loader, model, reg_criterion,
                      cls_criterion, epoch, args, writer)

        is_best = f1 > best_f1
        best_f1 = max(f1, best_f1)

        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            },
            ckpt_dir=ckpt_dir,
            filename='checkpoint_{}.pth.tar'.format(epoch),
            is_best=is_best)


def train(train_loader, model, reg_criterion, cls_criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Train Time', ':6.3f')
    data_time = AverageMeter('Train Data', ':6.3f')
    reg_losses = AverageMeter('Train Regression Loss', ':.4e')
    cls_losses = AverageMeter('Train Classification Loss', ':.4e')
    total_losses = AverageMeter('Train Total Loss', ':.4e')
    accuracy = AverageMeter('Train Accuracy', ':6.2f')
    precision = AverageMeter('Train Precision', ':6.2f')
    recall = AverageMeter('Train Recall', ':6.2f')
    f1_score = AverageMeter('Train F1 Score', ':6.2f')

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, reg_losses, cls_losses,
            accuracy, precision, recall, f1_score],
        prefix="Train Epoch: [{}]".format(epoch),
        writer=writer)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (obj_pos, arm_action, gripper_action) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        obj_pos = obj_pos.cuda(args.gpu, non_blocking=True)
        arm_action = arm_action.cuda(args.gpu, non_blocking=True)
        gripper_action = gripper_action.cuda(args.gpu, non_blocking=True)

        # compute output
        arm_action_pred, gripper_action_pred = model(obj_pos)
        reg_loss = reg_criterion(arm_action_pred, arm_action)
        cls_loss = cls_criterion(gripper_action_pred, gripper_action)
        loss = args.reg_loss_weight * reg_loss + args.cls_loss_weight * cls_loss

        # measure accuracy and record loss
        accu, prec, recl, f1 = get_cls_metrics(
            gripper_action_pred, gripper_action)
        reg_losses.update(reg_loss.item(), obj_pos.size(0))
        cls_losses.update(cls_loss.item(), obj_pos.size(0))
        total_losses.update(loss.item(), obj_pos.size(0))
        accuracy.update(accu.item(), obj_pos.size(0))
        precision.update(prec.item(), obj_pos.size(0))
        recall.update(recl.item(), obj_pos.size(0))
        f1_score.update(f1.item(), obj_pos.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    progress.log_tensorboard(epoch)


def validate(val_loader, model, reg_criterion, cls_criterion, epoch, args, writer=None):
    batch_time = AverageMeter('Val Time', ':6.3f')
    data_time = AverageMeter('Val Data', ':6.3f')
    reg_losses = AverageMeter('Val Regression Loss', ':.4e')
    cls_losses = AverageMeter('Val Classification Loss', ':.4e')
    total_losses = AverageMeter('Val Total Loss', ':.4e')
    accuracy = AverageMeter('Val Accuracy', ':6.2f')
    precision = AverageMeter('Val Precision', ':6.2f')
    recall = AverageMeter('Val Recall', ':6.2f')
    f1_score = AverageMeter('Val F1 Score', ':6.2f')

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, data_time, reg_losses, cls_losses,
            accuracy, precision, recall, f1_score],
        prefix="Val Epoch: [{}]".format(epoch),
        writer=writer)

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (obj_pos, arm_action, gripper_action) in enumerate(val_loader):
            obj_pos = obj_pos.cuda(args.gpu, non_blocking=True)
            arm_action = arm_action.cuda(args.gpu, non_blocking=True)
            gripper_action = gripper_action.cuda(
                args.gpu, non_blocking=True)

            # compute output
            arm_action_pred, gripper_action_pred = model(obj_pos)
            reg_loss = reg_criterion(arm_action_pred, arm_action)
            cls_loss = cls_criterion(gripper_action_pred, gripper_action)
            loss = args.reg_loss_weight * reg_loss + args.cls_loss_weight * cls_loss

            # measure accuracy and record loss
            accu, prec, recl, f1 = get_cls_metrics(
                gripper_action_pred, gripper_action)
            reg_losses.update(reg_loss.item(), obj_pos.size(0))
            cls_losses.update(cls_loss.item(), obj_pos.size(0))
            total_losses.update(loss.item(), obj_pos.size(0))
            accuracy.update(accu.item(), obj_pos.size(0))
            precision.update(prec.item(), obj_pos.size(0))
            recall.update(recl.item(), obj_pos.size(0))
            f1_score.update(f1.item(), obj_pos.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

    progress.log_tensorboard(epoch)
    return f1_score.avg


if __name__ == '__main__':
    main()
