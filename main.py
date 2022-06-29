import os

import numpy as np
from sklearn.model_selection import KFold
import segmentation_models_pytorch
import torch

import dataset
import models
import constant
import utils

import random

random.seed(constant.RANDOM_STATE)
np.random.seed(constant.RANDOM_STATE)
torch.manual_seed(constant.RANDOM_STATE)
torch.cuda.manual_seed(constant.RANDOM_STATE)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

init_model = 'deeplab_pp'

# model = torch.load(''.join([init_model, '.pth']))
model = models.model_dict[init_model]
optimizer = torch.optim.AdamW(model.parameters(), lr=constant.LEARNING_RATE)
loss_bce = torch.nn.BCEWithLogitsLoss()
loss_tversky = segmentation_models_pytorch.losses.TverskyLoss(mode='multilabel', log_loss=True, alpha=.5, beta=.5)
loss_focal = segmentation_models_pytorch.losses.FocalLoss(mode='multilabel')


def loss(preds, mask):
    """Combo loss function"""
    bce = loss_bce(preds, mask)
    tversky = loss_tversky(preds, mask)
    focal = loss_focal(preds, mask)
    return bce + tversky + focal


splits = KFold(n_splits=constant.K_FOLDS, shuffle=True, random_state=constant.RANDOM_STATE)

for fold, (train_index, valid_index) in enumerate(splits.split(dataset.df)):
    print('Fold {}/{}:'.format(fold, constant.K_FOLDS - 1), flush=True)
    # Train/Valid sets of data
    train = dataset.df.iloc[train_index]
    valid = dataset.df.iloc[valid_index]
    # Datasets
    train_dataset = utils.MetalDataset(
        train,
        transform=dataset.train_transform
    )
    valid_dataset = utils.MetalDataset(
        valid,
        transform=dataset.valid_transform
    )
    # Dataloaders
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=constant.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=constant.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        pin_memory=True
    )
    # Train model for current fold
    model = utils.train_model(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        loss=loss,
        optimizer=optimizer,
        num_epochs=constant.EPOCH,
        fold=fold
    )
