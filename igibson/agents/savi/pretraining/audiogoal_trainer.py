import os
import time
import logging
import copy
import shutil

import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch
import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from igibson.agents.savi.pretraining.audiogoal_predictor import AudioGoalPredictor
from igibson.agents.savi.pretraining.audiogoal_dataset import AudioGoalDataset

from igibson.agents.savi.utils.dataset import SCENE_SPLITS
from igibson.utils.utils import parse_config


class AudioGoalPredictorTrainer:
    def __init__(self, config, model_dir, predict_label, predict_location):
        self.config = config
        self.model_dir = model_dir
        self.device = (torch.device("cuda", 0))

        self.batch_size = 90
        self.num_worker = 4
        self.lr = 1e-3
        self.weight_decay = None
        self.num_epoch = 50
        self.audiogoal_predictor = AudioGoalPredictor(predict_label=predict_label,
                                                      predict_location=predict_location).to(device=self.device)
        self.predict_label = predict_label
        self.predict_location = predict_location
        summary(self.audiogoal_predictor.predictor, (2, 65, 69), device='cuda')

    def run(self, splits, writer=None):
        datasets = dict()
        dataloaders = dict()
        dataset_sizes = dict()
        for split in splits:
            scenes = SCENE_SPLITS[split]
            datasets[split] = AudioGoalDataset(
                scenes=scenes,
                split=split,
                use_polar_coordinates=False,
                use_cache=True
            )
            dataloaders[split] = DataLoader(dataset=datasets[split],
                                            batch_size=self.batch_size,
                                            shuffle=False,
                                            pin_memory=True,
                                            num_workers=self.num_worker,
                                            sampler=None,)

            dataset_sizes[split] = len(datasets[split])
            print('{} has {} samples'.format(split.upper(), dataset_sizes[split]))

        regressor_criterion = nn.MSELoss().to(device=self.device)
        classifier_criterion = nn.CrossEntropyLoss().to(device=self.device)
        model = self.audiogoal_predictor
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

        # training params
        since = time.time()
        best_acc = 0
        best_model_wts = None
        num_epoch = self.num_epoch if 'train' in splits else 1
        for epoch in range(num_epoch):
            logging.info('-' * 10)
            logging.info('Epoch {}/{}'.format(epoch, num_epoch))
            print('-' * 10)
            print('Epoch {}/{}'.format(epoch, num_epoch))

            # Each epoch has a training and validation phase
            for split in splits:
                print("split:", split)
                if split == 'train':
                    self.audiogoal_predictor.train()  # Set model to training mode
                else:
                    self.audiogoal_predictor.eval()  # Set model to evaluate mode

                running_total_loss = 0.0
                running_regressor_loss = 0.0
                running_classifier_loss = 0.0
                running_regressor_corrects = 0
                running_classifier_corrects = 0
                # Iterating over data once is one epoch
                for i, data in enumerate(tqdm(dataloaders[split])):
                    # get the inputs
                    inputs, gts = data
                    inputs = [x.to(device=self.device, dtype=torch.float) for x in inputs]
                    gts = gts.to(device=self.device, dtype=torch.float)
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    # forward
                    predicts = model({input_type: x for input_type, x in zip(['spectrogram'], inputs)})
                    if self.predict_label and self.predict_location:
                        classifier_loss = classifier_criterion(predicts[:, :-2], gts[:, 0].long())
                        regressor_loss = regressor_criterion(predicts[:, -2:], gts[:, -2:])
                    elif self.predict_label:
                        classifier_loss = classifier_criterion(predicts, gts[:, 0].long())
                    elif self.predict_location:
                        regressor_loss = regressor_criterion(predicts, gts[:, -2:])
                        classifier_loss = torch.tensor([0], device=self.device)
                    else:
                        raise ValueError('Must predict one item.')
                    loss = classifier_loss #+ regressor_loss
                    if split == 'train':
                        loss.backward()
                        optimizer.step()
                    running_total_loss += loss.item() * gts.size(0)
                    running_classifier_loss += classifier_loss.item() * gts.size(0)

                    pred_x = np.round(predicts.cpu().detach().numpy())
                    pred_y = np.round(predicts.cpu().detach().numpy())
                    gt_x = np.round(gts.cpu().numpy())
                    gt_y = np.round(gts.cpu().numpy())

                    # hard accuracy
                    if self.predict_label and self.predict_location:
                        running_regressor_corrects += np.sum(np.bitwise_and(
                            pred_x[:, -2] == gt_x[:, -2], pred_y[:, -1] == gt_y[:, -1]))
                        running_classifier_corrects += torch.sum(
                            torch.argmax(torch.abs(predicts[:, :-2]), dim=1) == gts[:, 0]).item()
                    elif self.predict_label:
                        running_classifier_corrects += torch.sum(
                            torch.argmax(torch.abs(predicts), dim=1) == gts[:, 0]).item()
                    elif self.predict_location:
                        running_regressor_corrects += np.sum(np.bitwise_and(
                            pred_x[:, 0] == gt_x[:, -2], pred_y[:, 1] == gt_y[:, -1]))
                        running_classifier_corrects = 0

                epoch_total_loss = running_total_loss / dataset_sizes[split]
                epoch_classifier_loss = running_classifier_loss / dataset_sizes[split]
                epoch_classifier_acc = running_classifier_corrects / dataset_sizes[split]
                if writer is not None:
                    writer.add_scalar(f'Loss/{split}_total', epoch_total_loss, epoch)
                    writer.add_scalar(f'Loss/{split}_classifier', epoch_classifier_loss, epoch)
                    writer.add_scalar(f'Accuracy/{split}_classifier', epoch_classifier_acc, epoch)
                logging.info(f'{split.upper()} Total loss: {epoch_total_loss:.4f}, '
                             f'label loss: {epoch_classifier_loss:.4f},' #  xy loss: {epoch_regressor_loss},
                             f' label acc: {epoch_classifier_acc:.4f},') #  xy acc: {epoch_regressor_acc}
                print(f'{split.upper()} Total loss: {epoch_total_loss:.4f}, '
                             f'label loss: {epoch_classifier_loss:.4f},' #  xy loss: {epoch_regressor_loss},
                             f' label acc: {epoch_classifier_acc:.4f},') # xy acc: {epoch_regressor_acc}
                # deep copy the model
                if self.predict_label and self.predict_location:
                    target_acc = epoch_regressor_acc + epoch_classifier_acc
                elif self.predict_location:
                    target_acc = epoch_regressor_acc
                else:
                    target_acc = epoch_classifier_acc

                if split == 'val' and target_acc > best_acc:
                    best_acc = target_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    self.save_checkpoint(f"ckpt.{epoch}.pth")

        self.save_checkpoint(f"best_val.pth", checkpoint={"audiogoal_predictor": best_model_wts})

        time_elapsed = time.time() - since
        logging.info('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        logging.info('Best val acc: {:4f}'.format(best_acc))

        if best_model_wts is not None:
            model.load_state_dict(best_model_wts)

    def save_checkpoint(self, ckpt_path, checkpoint=None):
        if checkpoint is None:
            checkpoint = {
                "audiogoal_predictor": self.audiogoal_predictor.state_dict(),
            }
        torch.save(
            checkpoint, os.path.join(self.model_dir, ckpt_path)
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        # required=True,
        default='train',
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--model-dir",
        default='data/models/audiogoal_predictor',
        help="Modify config options from command line",
    )
    parser.add_argument(
        "--overwrite",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--predict-location",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    parser.add_argument(
        "--predict-label",
        default=False,
        action='store_true',
        help="Modify config options from command line"
    )
    args = parser.parse_args()
    config = parse_config('pretraining/config/savi.yaml')
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s, %(levelname)s: %(message)s',
                        datefmt="%Y-%m-%d %H:%M:%S")

    log_dir = os.path.join(args.model_dir, 'tb')
    if args.run_type == 'train' and os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    audiogoal_predictor_trainer = AudioGoalPredictorTrainer(config, args.model_dir, predict_location=args.predict_location,
                                                            predict_label=args.predict_label)

    if args.run_type == 'train':
        writer = SummaryWriter(log_dir=log_dir)
        audiogoal_predictor_trainer.run(['train', 'val'], writer)
    else:
        ckpt = torch.load(os.path.join(args.model_dir, 'val_best.pth'))
        audiogoal_predictor_trainer.audiogoal_predictor.load_state_dict(ckpt['audiogoal_predictor'])
        audiogoal_predictor_trainer.run(['test'])


if __name__ == '__main__':
    main()

