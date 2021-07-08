import torch
import torch.nn as nn
import torchvision

def get_mean_std(loader):
    ch_sum, ch_squared_sum, count_of_batches = 0, 0, 0
    
    for data, _ in loader:
        ch_sum += torch.mean(data, dim=[0, 2, 3])
        ch_squared_sum += torch.mean(data**2, dim=[0, 2, 3])
        count_of_batches += 1

    mean = ch_sum / count_of_batches 
    std = (ch_squared_sum / count_of_batches - mean**2)**0.5

    return mean, std 


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice


def soft_dice(*, y_true, y_pred):
    eps = 1e-15
    y_pred = y_pred.contiguous().view(y_pred.numel())
    y_true = y_true.contiguous().view(y_true.numel())
    intersection = (y_pred * y_true).sum(0)
    scores = 2. * (intersection + eps) / (y_pred.sum(0) + y_true.sum(0) + eps)
    score = scores.sum() / scores.numel()
    return torch.clamp(score, 0., 1.)


def hard_dice(*, y_true, y_pred):
    y_pred = torch.round(y_pred)
    return soft_dice(y_true=y_true, y_pred=y_pred)


def accuracy(y_true, y_pred, thr=0.5):
    num_correct = 0
    num_pixels = 0
    
    y_pred = (y_pred > thr).float()
    num_correct += (y_true == y_pred).sum()
    num_pixels += torch.numel(y_pred)
    
    return num_correct/num_pixels*100

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