import ast
import os
import glob
import tqdm

import cv2
import numpy as np
import pandas as pd
from monai.metrics.utils import get_mask_edges, get_surface_distance
import torch

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

sns.set_style('darkgrid')

import constant

print(f'Torch version: {torch.__version__}')
print(f'Cuda?: {torch.cuda.is_available()}\n')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


def data_spliter(dataframe):
    df = pd.read_csv(dataframe, sep=' ', index_col=0)
    train, valid = train_test_split(
        df,
        train_size=.8,
        random_state=42,
        shuffle=True
    )
    return train, valid, df


def load_img(path, size=[320, 384]):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    shape0 = np.array(img.shape[:2])
    resize = np.array(size)
    if np.any(shape0 != resize):
        diff = resize - shape0
        pad0 = diff[0]
        pad1 = diff[1]
        pad_y = [pad0 // 2, pad0 // 2 + pad0 % 2]
        pad_x = [pad1 // 2, pad1 // 2 + pad1 % 2]
        img = np.pad(img, [pad_y, pad_x])
        img = img.reshape(resize)
    return img, shape0


def load_imgs(img_paths, size=[320, 384]):
    imgs = np.zeros((*size, len(img_paths)))
    for i in range(len(img_paths)):
        if i == 0:
            img, shape0 = load_img(img_paths[i], size=size)
        else:
            img, _ = load_img(img_paths[i], size=size)
        imgs[..., i] += img
    return imgs, shape0


class MetalDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None):
        self.images_files = data['image_paths'].tolist()
        self.masks_files = data['mask_path'].tolist()
        self.transform = transform

    def __len__(self):
        return len(self.images_files)

    def __getitem__(self, index):
        # Select on image-mask couple
        image_path = ast.literal_eval((self.images_files[index]))
        mask_path = self.masks_files[index]
        # Image 2.5D load
        image, _ = load_imgs(image_path)
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
        for i in range(image.shape[-1]):
            image[i] = cv2.equalizeHist(image[i])
        # Maks processing
        mask = np.load(mask_path) / 255
        mask = mask.astype(np.uint8)
        # Augmentation
        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']
        return image, mask


def batch_image_mask_show(dataloader, number_of_images=5, initial_index=0):
    """Plot samples after augmentation"""
    images, masks = next(iter(dataloader))
    for tensor in [images, masks]:
        if tensor is masks:
            tensor = tensor.numpy().transpose(0, 2, 3, 1)
            tensor = tensor * 255
        else:
            tensor = tensor.numpy().transpose(0, 2, 3, 1)
            tensor = constant.STD * tensor + constant.MEAN
        fig = plt.figure(figsize=(12, 7))
        for i in range(number_of_images):
            fig.add_subplot(1, number_of_images + 1, i + 1)
            plt.imshow(tensor[i + initial_index], cmap='icefire')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
    plt.show()


def dice_coef(y_true, y_pred, thr=constant.THRESHOLD, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2 * inter + epsilon) / (den + epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=constant.THRESHOLD, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true * y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true * y_pred).sum(dim=dim)
    iou = ((inter + epsilon) / (union + epsilon)).mean(dim=(1, 0))
    return iou


def train_model(model, train_dataloader, valid_dataloader, loss, optimizer, num_epochs, fold):
    """Train and Validate Model"""
    train_loss, valid_loss = [], []

    model = model.to(device)
    min_loss = .2
    for epoch in range(num_epochs):
        print('Epoch {}/{}:'.format(epoch, num_epochs - 1), flush=True)
        # Each epoch has a training and validation phase
        for phase in ['Train', 'Valid']:
            if phase == 'Train':
                dataloader = train_dataloader
                model.train()  # Set model to training mode
            else:
                dataloader = valid_dataloader
                model.eval()  # Set model to evaluate mode
            running_loss = 0.
            running_dice = 0.
            running_iou = 0.
            # Iterate over data.
            for image, mask in tqdm.tqdm(dataloader):
                image = image.to(device, dtype=torch.float32)
                mask = mask.to(device, dtype=torch.float32)
                optimizer.zero_grad()
                # forward and backward
                with torch.set_grad_enabled(phase == 'Train'):
                    preds = model(image)
                    loss_value = loss(preds, mask)
                    # Compute metrics
                    dice = dice_coef(mask, preds).cpu().detach().numpy()
                    iou = iou_coef(mask, preds).cpu().detach().numpy()
                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss_value.backward()
                        optimizer.step()
                # statistics
                running_loss += loss_value.item()
                running_dice += dice
                running_iou += iou
            # Average values along one epoch
            epoch_loss = running_loss / len(dataloader)
            epoch_dice = running_dice / len(dataloader)
            epoch_iou = running_iou / len(dataloader)
            # Checkpoint
            if epoch_loss < min_loss and phase != 'Train':
                min_loss = epoch_loss
                model = model.cpu()
                torch.save(model, f'checkpoint\\model_fold_{fold}_loss_{min_loss:.3f}.pth')
                model = model.to(device)
            # Epoch final metric
            if phase == 'Train':
                train_loss.append(epoch_loss)
            else:
                valid_loss.append(epoch_loss)
            # Show results on current step
            print('{} Loss: {:.4f} Dice: {:.4f} IOU: {:.4f}'.format(
                phase, epoch_loss, epoch_dice, epoch_iou
            ), flush=True)

    return model
