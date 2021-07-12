import torch
import torch.nn as nn
import torchvision
import skimage.io
import numpy as np


def get_mean_std(loader):
    ch_sum, ch_squared_sum, count_of_batches = 0, 0, 0
    
    for data, _ in loader:
        ch_sum += torch.mean(data, dim=[0, 2, 3])
        ch_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        count_of_batches += 1

    mean = ch_sum / count_of_batches 
    std = (ch_squared_sum / count_of_batches - mean**2)**0.5

    return mean, std 


def soft_dice(*, y_true, y_pred):
    eps = 1e-15
    y_pred = y_pred.contiguous().view(y_pred.numel())
    y_true = y_true.contiguous().view(y_true.numel())
    intersection = (y_pred * y_true).sum(0)
    scores = 2. * (intersection + eps) / (y_pred.sum(0) + y_true.sum(0) + eps)
    score = scores.sum() / scores.numel()
    
    return torch.clamp(score, 0., 1.)


def hard_dice(*, y_true, y_pred, thr=0.5):
    y_pred = (y_pred > thr).float()
    return soft_dice(y_true=y_true, y_pred=y_pred)


def accuracy(y_true, y_pred, thr=0.5):
    num_correct = 0
    num_pixels = 0
    
    y_pred = (y_pred > thr).float()
    num_correct += (y_true == y_pred).sum()
    num_pixels += torch.numel(y_pred)
    
    return num_correct/num_pixels*100


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, target):
        return 1 - soft_dice(y_true=target, y_pred=torch.sigmoid(inputs))


class BCEDiceLoss(nn.Module):
    def __init__(self, dice_weight=0.5):
        super().__init__()
        self._dice = DiceLoss()
        self._dice_weight = dice_weight

    def forward(self, inputs, target):
        return (1 - self._dice_weight) * nn.BCEWithLogitsLoss()(inputs, target) + \
            self._dice_weight * self._dice(inputs, target)


def read_mask(mask_name):
    IMG_HEIGHT = 256
    mask = (skimage.io.imread(mask_name)[:,:,-1]==255).astype(np.uint8)*255
    mask = skimage.transform.rescale(mask, IMG_HEIGHT * 1. / mask.shape[0], order=0, preserve_range=True)
    mask = (mask > 0).astype(np.uint8)

    return mask



def read_image(img_name, mask_name=None):
    IMG_HEIGHT = 256
    im = skimage.io.imread(img_name)
    im = skimage.transform.rescale(im, IMG_HEIGHT * 1. / im.shape[0], multichannel=True)
    im = skimage.img_as_ubyte(im)
    if mask_name is not None:
        mask = read_mask(mask_name)
        return im, mask
    return im


def make_blending(img_path, mask_path, alpha=0.5):
    img, mask = read_image(img_path, mask_path)
    colors = np.array([[0,0,0], [0,255,0]], np.uint8)
    return (img*alpha + colors[mask.astype(np.int32)]*(1. - alpha)).astype(np.uint8)


def save_predictions_as_imgs(loader, model, thr=0.5, folder="saved_images/", device='cuda'):
    model.eval()
    for idx, data in enumerate(loader):
        x = data['image'].to(device=device)
        y = data['mask']
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > thr).float()
        x = x.float() / 255
        torchvision.utils.save_image(x.data.cpu(), f"{folder}/orig_{idx}.png")
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idx}.png")

    model.train()