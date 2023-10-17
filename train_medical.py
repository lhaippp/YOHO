import os

import numpy as np
import torch
import argparse
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.unet import Unet
from nets.unet_training import weights_init
from utils.callbacks import LossHistory
from utils.dataloader_medical import UnetDataset, unet_dataset_collate
from utils.utils_fit import fit_one_epoch_no_val

parser = argparse.ArgumentParser()
parser.add_argument("--png_name", default="dummy", help="Input image name")
args = parser.parse_args()

if __name__ == "__main__":
    # Whether to use Cuda
    Cuda = True

    # Number of classifications required
    num_classes = 2

    # Backbone
    backbone = "resnet34"

    # Whether or not to use the pre-training weights of the backbone network, here the weights of the backbone are used, and are therefore loaded at the time of model construction.
    # if model_path is set, then the weights of the backbone do not need to be loaded and the pretrained values are meaningless
    # if model_path is not set, pretrained = True. At this point, only the backbone is loaded to start training
    # if model_path is not set, pretrained = False, Freeze_Train = False. At this point training starts from 0 and there is no process of freezing the backbone.
    pretrained = True

    model_path = ""

    # The size of the input image
    input_shape = [256, 256]

    # The training is divided into two phases, a freezing phase and a thawing phase.
    # The lack of memory has nothing to do with the size of the dataset, please adjust the batch_size.

    # Freeze phase training parameters
    # The backbone of the model is frozen at this point and the feature extraction network is not changed.
    # Less memory is used, only fine-tuning of the network is done
    Init_Epoch = 0
    Freeze_Epoch = 20
    Freeze_batch_size = 32
    Freeze_lr = 1e-3

    # Unfreeze phase training parameters
    # At this point the backbone of the model is not frozen and the feature extraction network changes
    # A larger amount of memory is used and all the parameters of the network are changed
    # ----------------------------------------------------#
    UnFreeze_Epoch = 30
    Unfreeze_batch_size = 32
    Unfreeze_lr = 3e-5

    # The path of the dataset
    VOCdevkit_path = "Medical_Datasets"

    dice_loss = False

    focal_loss = False

    # Whether to assign different loss weights to different categories
    cls_weights = np.array([1, 1], np.float32)

    # Whether or not to freeze the training, the default is to freeze the main training first and then unfreeze the training.
    Freeze_Train = True

    num_workers = 2
    # Whether to consider edge
    comb_edge: bool = True

    # The name of the dataset, related to the save of the weights file
    dataset_name = "EEC"

    model = Unet(
        num_classes=num_classes, pretrained=pretrained, backbone=backbone
    ).train()
    if not pretrained:
        weights_init(model)
    if model_path != "":
        print("Load weights {}.".format(model_path))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if np.shape(model_dict[k]) == np.shape(v)
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    model_train = model.train()

    if Cuda:
        model_train = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model_train = model_train.cuda()

        loss_history = LossHistory("logs/", val_loss_flag=False)

    # Read the corresponding txt of the dataset
    with open(
        os.path.join(VOCdevkit_path, "ImageSets/Segmentation/train.txt"), "r"
    ) as f:
        train_lines = f.readlines()

    # Not enough memory, please reduce Batch_size
    if True:
        batch_size = Freeze_batch_size
        lr = Freeze_lr
        start_epoch = Init_Epoch
        end_epoch = Freeze_Epoch

        epoch_step = len(train_lines) // batch_size

        if epoch_step == 0:
            raise ValueError(
                "The dataset is too small for training, please expand the dataset."
            )

        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = UnetDataset(
            train_lines, input_shape, num_classes, True, VOCdevkit_path
        )
        gen = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=unet_dataset_collate,
        )

        # Freeze trainning
        if Freeze_Train:
            model.freeze_backbone()

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch_no_val(
                model_train,
                model,
                loss_history,
                optimizer,
                epoch,
                epoch_step,
                gen,
                end_epoch,
                Cuda,
                dice_loss,
                focal_loss,
                cls_weights,
                num_classes,
                comb_edge,
                args.png_name,
                dataset_name,
            )
            lr_scheduler.step()

    if True:
        batch_size = Unfreeze_batch_size
        lr = Unfreeze_lr
        start_epoch = Freeze_Epoch
        end_epoch = UnFreeze_Epoch

        epoch_step = len(train_lines) // batch_size

        if epoch_step == 0:
            raise ValueError(
                "The dataset is too small for training, please expand the dataset."
            )
        optimizer = optim.Adam(model_train.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.96)

        train_dataset = UnetDataset(
            train_lines, input_shape, num_classes, True, VOCdevkit_path
        )
        gen = DataLoader(
            train_dataset,
            shuffle=True,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=unet_dataset_collate,
        )

        if Freeze_Train:
            model.unfreeze_backbone()

        for epoch in range(start_epoch, end_epoch):
            fit_one_epoch_no_val(
                model_train,
                model,
                loss_history,
                optimizer,
                epoch,
                epoch_step,
                gen,
                end_epoch,
                Cuda,
                dice_loss,
                focal_loss,
                cls_weights,
                num_classes,
                comb_edge,
                args.png_name,
                dataset_name,
            )
            lr_scheduler.step()
