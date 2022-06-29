import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

import constant
import utils

_, _, df = utils.data_spliter(constant.DATAFRAME)
# Augmentation
train_transform = A.Compose([
    # Non destructive
    A.OneOf([
        A.HorizontalFlip(p=1.),
        A.VerticalFlip(p=1.)
    ], p=.2),
    # Non-rigid block
    A.OneOf([
        A.ElasticTransform(p=1.),
        A.GridDistortion(p=1.),
        A.OpticalDistortion(p=1.),
    ], p=.2),
    # Blur block
    A.OneOf([
        A.MotionBlur(blur_limit=3, p=1.),
        A.MedianBlur(blur_limit=3, p=1.),
        A.Blur(blur_limit=3, p=1.),
        A.GaussianBlur(p=1.),
    ], p=.2),
    # Noise block
    A.OneOf([
        # A.ISONoise(p=1.),
        A.GaussNoise(p=1.),
        A.MultiplicativeNoise(p=1.)
    ], p=.2),
    # Contrast block
    A.OneOf([
        A.RandomBrightnessContrast(p=1.),
        A.RandomGamma(p=1.),
        A.CLAHE(p=1.)
    ], p=.2),
    A.Normalize(mean=constant.MEAN, std=constant.STD, p=1.),
    ToTensorV2(transpose_mask=True)
])

valid_transform = A.Compose([
    A.Normalize(mean=constant.MEAN, std=constant.STD, p=1.),
    ToTensorV2(transpose_mask=True)
])
# Datasets
# train_dataset = utils.MetalDataset(
#     train,
#     transform=train_transform
# )
# valid_dataset = utils.MetalDataset(
#     valid,
#     transform=valid_transform
# )
# full_dataset = utils.MetalDataset(
#     data=df,
#     transform=train_transform
# )
# Check data shape for image-mask
# print(f'Image shape:\n{list(train_dataset[0][0].shape)}')
# print(f'Mask shape:\n{list(train_dataset[0][1].shape)}\n')
# # Check train-valid size
# print(f'Train dataset length: {train_dataset.__len__()}')
# print(f'Valid dataset length: {valid_dataset.__len__()}')
# print(f'Full dataset length: {full_dataset.__len__()}\n')
# Dataloaders
# train_dataloader = torch.utils.data.DataLoader(
#     train_dataset,
#     batch_size=constant.BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     pin_memory=True
# )
# valid_dataloader = torch.utils.data.DataLoader(
#     valid_dataset,
#     batch_size=constant.BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     pin_memory=True
# )
# full_dataloader = torch.utils.data.DataLoader(
#     full_dataset,
#     batch_size=constant.BATCH_SIZE,
#     shuffle=True,
#     drop_last=True,
#     pin_memory=True
# )
# utils.batch_image_mask_show(dataloader=valid_dataloader)
