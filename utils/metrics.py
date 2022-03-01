import torch
import torch.nn as nn
import torch.nn.functional as F


class BCEDiceLoss(nn.Module):   # done

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        dice = 1. - dice_coef(input, target)
        return bce + dice
        # bce = F.binary_cross_entropy_with_logits(input, target)
        # smooth = 1e-5
        # input = torch.sigmoid(input)
        # num = target.size(0)
        # input = input.view(num, -1)
        # target = target.view(num, -1)
        # # target = torch.unsqueeze(target, 0)
        # intersection = (input * target)
        # dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        # dice = 1 - dice.sum() / num
        # return 0.5 * bce + dice


def iou_score(output, target):  # TODO: 跟tf版不一样
    smooth = 1e-5
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return ((intersection + smooth) / (union + smooth)).cpu().item()


def dice_coef(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return ((2. * intersection +smooth) / (output.sum() + target.sum() + smooth)).cpu().item()
