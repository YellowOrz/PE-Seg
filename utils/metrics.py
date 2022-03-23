import torch.nn.functional as F


def dice_coef(output, target):
    smooth = 1e-5
    intersection = (output * target).sum()
    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


def dice_loss(output, target):
    return 1.0 - dice_coef(output, target)


def bce_dice_loss(output, target):
    bce_loss = F.binary_cross_entropy_with_logits(output, target)
    dice_loss = 1. - dice_coef(output, target)
    return bce_loss + dice_loss


def iou_score(output, target):  # TODO: 跟tf版不一样
    smooth = 1e-5
    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return ((intersection + smooth) / (union + smooth)).item()

def tversky_focal_loss(output, target):
        alpha=0.7
        gama=0.75
        beta = 1 - alpha
        smooth = 1e-5
        output = output.view(-1)
        target = target.view(-1)
        TP = (output * target).sum()
        FN = ((1 - output) * target).sum()
        FP = ((1 - target) * output).sum()
        TI = (TP + smooth) / (TP + alpha * FN + beta * FP + smooth)
        return pow(1 - TI, gama)

class AverageMeter(object):

    def __init__(self):
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
