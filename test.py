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
from dataset import *
from train import *


print('-' * 20)
for key in config.keys():
    print('%s: %s' % (key, str(config[key])))
print('-' * 20)

cudnn.benchmark = True

# model = DoubleUNet(config['num_classes'], config['input_classes'])
model = DoubleUNet()
model = model.cuda()

# Data loading code
img_ids = glob(os.path.join(config['dataset'], 'images', '*' + config['img_ext']))
img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]

_, val_img_ids = train_test_split(img_ids, test_size=0.2, random_state=41)

model.load_state_dict(torch.load('models/%s/model.pth' % config['name']))

model.eval()

val_transform = Compose([
    transforms.Resize(config['input_h'], config['input_w']),
    transforms.Normalize(),
])

val_dataset = Data(
    img_ids=val_img_ids,
    img_dir=os.path.join('inputs', config['dataset'], 'images'),
    mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
    img_ext=config['img_ext'],
    mask_ext=config['mask_ext'],
    num_classes=config['num_classes'],
    transforms=val_transform
)

val_loader = torch.utils.data.Dataloader(
    val_dataset,
    batch_size=config['batch_size'],
    shuffle=False,
    num_workers=config['num_workers'],
    drop_last=False
)

avg_meter = AverageMeter()

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

        dice_score = dice_coef(output, target)
        avg_meter.update(dice_score, input.size(0))

        output = torch.sigmoid(output).cpu().numpy()

        for i in range(len(output)):
            for c in range(config['num_classes']):
                cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                            (output[i, c] * 255).astype('uint8'))

print('Dice: %.4f' % avg_meter.avg)

torch.cuda.empty_cache()