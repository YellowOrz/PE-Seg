from tqdm import tqdm
import utils.data_loaders
import utils.data_transforms
from collections import OrderedDict
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
import argparse
from utils.metrics import *
from config import *
from DoubleUNet import *
from utils.netword_util import *
from double_unet import DoubleUnet

def str2bool(v):
    if v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        return argparse.ArgumentTypeError('Boolean Value Expected')


def count_params(model):
    return (sum(p.numel()) for p in model.parameters() if p.requires_grad)


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


def train(config, train_loader, model, criterion, optimizer):
    global dice_score
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice_coef': AverageMeter()}
    model.train()

    pbar = tqdm(total=len(train_loader))
    for input, target in train_loader:
        input, target = input.cuda(), target.cuda()

        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
            dice_score = dice_coef(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        avg_meters['loss'].update(loss.item(), input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        avg_meters['dice_coef'].update(dice_score, input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),
                               ('dice_coef', avg_meters['dice_coef'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),
                        ('dice_coef', avg_meters['dice_coef'].avg)])


def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(), 'iou': AverageMeter(), 'dice_coef': AverageMeter()}

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target in val_loader:
            input = input.cuda()
            target = target.cuda()

            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)

            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
                dice_score = dice_coef(output, target)

            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            avg_meters['dice_coef'].update(dice_score, input.size(0))

            postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),
                                   ('dice_coef', avg_meters['dice_coef'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()

    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),
                        ('dice_coef', avg_meters['dice_coef'].avg)])


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu']
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%swDS' % (config['arch'])
        else:
            config['name'] = '%swoDS' % (config['arch'])
    os.makedirs('models/%s' % config['name'], exist_ok=True)

    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)

    criterion = nn.BCEWithLogitsLoss() if config['loss'] == 'BCEWithLogitsLoss' else BCEDiceLoss()

    cudnn.benchmark = True

    # model = DoubleUNet(config['num_classes'],config['input_channels'])
    # import torchvision.models as models
    # model = DoubleUnet(models.vgg19_bn()).cuda()
    model = DoubleUNet().cuda()
    # model.apply(init_weights_kaiming)

    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(params, lr=config['lr'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError

    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], patience=config['patience'],
                                                   verbose=True, min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(e) for e in config['milestones'].split(',')],
                                             gamma=config['gammma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError

    train_transforms = utils.data_transforms.Compose([
        utils.data_transforms.RandomCrop(config['input_h'], config['input_w']),  # TODO: crop or resize?
        # utils.data_transforms.ColorJitter(config['color_jitter']),
        utils.data_transforms.Normalize(mean=0, std=255),
        utils.data_transforms.RandomVerticalFlip(),
        utils.data_transforms.RandomHorizontalFlip(),
        # utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
        utils.data_transforms.ToTensor(),
    ])

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(mean=0, std=255),
        utils.data_transforms.ToTensor(),
    ])
    # train_transform = Compose([transforms.RandomRotate90(), transforms.Flip(),
    #                            OneOf([transforms.HueSaturationValue(), transforms.RandomBrightnessContrast()]),# , p=1?
    #                            transforms.Resize(config['input_h'], config['input_w']), transforms.Normalize()])
    #
    # test_transform = Compose([transforms.Resize(config['input_h'], config['input_w']),
    #                          transforms.Normalize(), ])
    #
    # train_dataset = Data(img_ids=train_img_ids, num_classes=config['num_classes'], transform=train_transform,
    #                      img_dir=os.path.join(config['dataset'], 'ISIC2018_Task1-2_Training_Input'),
    #                      mask_dir=os.path.join(config['dataset'], 'ISIC2018_Task1_Training_GroundTruth'))
    # val_dataset = Data(img_ids=val_img_ids, num_classes=config['num_classes'], transform=val_transform,
    #                    img_dir=os.path.join('inputs', config['dataset'], 'ISIC2018_Task1-2_Training_Input'),
    #                    mask_dir=os.path.join('inputs', config['dataset'], 'ISIC2018_Task1_Training_GroundTruth'))
    train_dataset = utils.data_loaders.DatasetSeq(config['dataset'], config['img_label'], config['mask_label'], 'train',
                                                  config['test_size'], config['num_classes'], train_transforms)
    test_dataset = utils.data_loaders.DatasetSeq(config['dataset'], config['img_label'], config['mask_label'], 'test',
                                                 config['test_size'], config['num_classes'], test_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                               num_workers=config['num_workers'], pin_memory=True,
                                               drop_last=True)      # drop_last必须，为了防止BN报错
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                                             num_workers=config['num_workers'], pin_memory=True)

    log = OrderedDict([('epoch', []),('lr', []),('loss', []),('iou', []),('dice', []),
                       ('val_loss', []),('val_iou', []),('val_dice', [])])

    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f -val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice_coef'],
                 val_log['loss'], val_log['iou'], val_log['dice_coef']))

        log['epoch'].append(epoch)
        log['lr'].append(config['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice_coef'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice_coef'])

        pd.DataFrame(log).to_csv('models/%s/log.csv' % config['name'], index=False)

        trigger += 1

        if val_log['dice_coef'] > best_dice:
            torch.save(model.state_dict(), 'models/%s/model.pth' % config['name'])
            best_dice = val_log['dice_coef']
            print("=> saved best model")
            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
