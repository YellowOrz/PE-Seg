import numpy as np
import os
import cv2
from tqdm import tqdm
from glob import glob
from sklearn.model_selection import train_test_split

from albumentations.core.composition import Compose, OneOf
from albumentations.augmentations import transforms

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset

from config import *
from model import *
from train import *


print('-' * 20)
for key in config.keys():
    print('%s: %s' % (key, str(config[key])))
print('-' * 20)

cudnn.benchmark = True

# model = DoubleUNet(config['num_classes'], config['input_classes'])
model = DoubleUNet()
model = model.cuda()
# DoubleUNetwoDS,convtranspose
model_name = 'DoubleUNet_tf'
model.load_state_dict(torch.load('models/%s/model.pth' % model_name))
model.eval()

# TODO: change dateload
test_transforms = utils.data_transforms.Compose([
    utils.data_transforms.Normalize(),
    utils.data_transforms.ToTensor(),
])

test_dataset = utils.data_loaders.DatasetSeq(config['dataset'], config['img_label'], config['mask_label'], 'test',
                                             config['test_size'], config['num_classes'], test_transforms)
val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False,
                                         num_workers=config['num_workers'], pin_memory=True)

avg_meters = {'dice_outs': AverageMeter(), 'iou_outs': AverageMeter(),
              'dice_out1': AverageMeter() , 'iou_out1':AverageMeter(),
              'dice_out2': AverageMeter() , 'iou_out2':AverageMeter(),}

for c in range(config['num_classes']):
    os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)

with torch.no_grad():
    for input, target, meta in tqdm(val_loader, total=len(val_loader)):
        input = input.cuda()
        target = target.cuda()

        # compute output
        if config['deep_supervision']:
            output = model(input)[-1]
        else:
            output = model(input)
        dice_score_outs = dice_coef(output, target)
        iou_outs = iou_score(output,target)
        # dice_score_outs = dice_coef((output[:,:1]+output[:,1:])/2, target)
        # iou_outs = iou_score((output[:,:1]+output[:,1:])/2,target)
        dice_out1 = dice_coef(output[:,:1,:,:],target[:,:1,:,:])
        iou_out1 = iou_score(output[:, :1, :, :], target[:, :1, :, :])
        dice_out2 = dice_coef(output[:, 1:, :, :], target[:, 1:, :, :])
        iou_out2 = iou_score(output[:, 1:, :, :], target[:, 1:, :, :])
        avg_meters['dice_outs'].update(dice_score_outs, input.size(0))
        avg_meters['iou_outs'].update(iou_outs, input.size(0))
        avg_meters['dice_out1'].update(dice_out1, input.size(0))
        avg_meters['iou_out1'].update(iou_out1, input.size(0))
        avg_meters['dice_out2'].update(dice_out2, input.size(0))
        avg_meters['iou_out2'].update(iou_out2, input.size(0))
        # output = torch.sigmoid(output).cpu().numpy()
        output = (output > 0.5).cpu().numpy()
        for c in range(2):
            cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta[0].split('/')[-1]),
                        (output[0, c] * 255).astype(np.uint8))

# print('Dice: %.4f   iou: %.4f' % avg_meter.avg,iou.avg)
print('dice_outs:',avg_meters['dice_outs'].avg,'   iou_outs:',avg_meters['iou_outs'].avg)
print('dice_out1:',avg_meters['dice_out1'].avg,'   iou_out1:',avg_meters['iou_out1'].avg)
print('dice_out2:',avg_meters['dice_out2'].avg,'   iou_out2:',avg_meters['iou_out2'].avg)
torch.cuda.empty_cache()