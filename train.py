from tqdm import tqdm
import utils.data_loaders
import utils.data_transforms
from collections import OrderedDict
import os
import pandas as pd
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms
from torch.utils.data import Dataset
from tensorboardX import SummaryWriter
from utils.metrics import *
from config import *
from DoubleUNet import *
from utils.netword_util import *
from datetime import datetime as dt


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
        avg_meters['dice_coef'].update(dice_score.item(), input.size(0))

        postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),
                               ('dice_coef', avg_meters['dice_coef'].avg)])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()

    save_img0 = torch.cat([input[0], output[0, :1].expand(3, -1, -1), target[0, :1].expand(3, -1, -1)], dim=-1)
    save_img1 = torch.cat([input[0], output[0, -1:].expand(3, -1, -1), target[0, :1].expand(3, -1, -1)], dim=-1)
    return OrderedDict(
        [('loss', avg_meters['loss'].avg), ('dice_coef', avg_meters['dice_coef'].avg), ('iou', avg_meters['iou'].avg),
         ('outimg0', save_img0.detach().cpu()), ('outimg1', save_img1.detach().cpu())])


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
            avg_meters['dice_coef'].update(dice_score.item(), input.size(0))

            postfix = OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),
                                   ('dice_coef', avg_meters['dice_coef'].avg)])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    save_img0 = torch.cat([input[0], output[0, :1].expand(3, -1, -1), target[0, :1].expand(3, -1, -1)], dim=-1)
    save_img1 = torch.cat([input[0], output[0, -1:].expand(3, -1, -1), target[0, :1].expand(3, -1, -1)], dim=-1)
    return OrderedDict([('loss', avg_meters['loss'].avg), ('iou', avg_meters['iou'].avg),
                        ('dice_coef', avg_meters['dice_coef'].avg), ('outimg0', save_img0), ('outimg1', save_img1)])


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
    output_folder = os.path.join(config['out_dir'], config['name'] + '_' + dt.now().strftime("%m-%d-%H-%M-%S"))
    print(output_folder)

    criterion = eval(config['loss'])
    writer = SummaryWriter(output_folder)
    cudnn.benchmark = True

    model = DoubleUNet().cuda()
    # model.apply(init_weights_kaiming)
    print_network(model)

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
        # utils.data_transforms.Resize([512,512]),
        utils.data_transforms.Normalize(),
        utils.data_transforms.RandomVerticalFlip(),
        utils.data_transforms.RandomHorizontalFlip(),
        # utils.data_transforms.RandomGaussianNoise(cfg.DATA.GAUSSIAN),
        utils.data_transforms.ToTensor(),
    ])

    test_transforms = utils.data_transforms.Compose([
        utils.data_transforms.Normalize(),
        utils.data_transforms.ToTensor(),
    ])

    train_dataset = utils.data_loaders.DatasetSeq(config['dataset'], config['img_label'], config['mask_label'], 'train',
                                                  config['test_size'], config['num_classes'], train_transforms)
    test_dataset = utils.data_loaders.DatasetSeq(config['dataset'], config['img_label'], config['mask_label'], 'test',
                                                 config['test_size'], config['num_classes'], test_transforms)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True,
                                               num_workers=config['num_workers'], pin_memory=True,
                                               drop_last=True)  # drop_last必须，为了防止BN报错
    val_loader = torch.utils.data.DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False,
                                             num_workers=config['num_workers'], pin_memory=True)
    log = OrderedDict([('epoch', []), ('lr', []), ('loss', []), ('iou', []), ('dice', []),
                       ('val_loss', []), ('val_iou', []), ('val_dice', [])])
    best_dice = 0
    trigger = 0
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))

        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        log2tensorboard(writer, train_log, epoch, 'tarin')
        # evaluate on validation set
        if epoch % config['val_frequency'] == 0:
            val_log = validate(config, val_loader, model, criterion)
        else:
            val_log = val_log
        log2tensorboard(writer, val_log, epoch, 'val')

        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        writer.add_scalar('PE_Seg/lr', optimizer.state_dict()['param_groups'][0]['lr'], epoch)

        print('loss %.4f - iou %.4f - dice %.4f - val_loss %.4f - val_iou %.4f -val_dice %.4f'
              % (train_log['loss'], train_log['iou'], train_log['dice_coef'],
                 val_log['loss'], val_log['iou'], val_log['dice_coef']))

        log['epoch'].append(epoch)
        log['lr'].append(optimizer.state_dict()['param_groups'][0]['lr'])
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['dice'].append(train_log['dice_coef'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        log['val_dice'].append(val_log['dice_coef'])

        pd.DataFrame(log).to_csv(os.path.join(output_folder, 'log.csv'.format(best_dice)), index=False)

        trigger += 1

        if val_log['dice_coef'] > best_dice:
            best_dice = val_log['dice_coef']
            torch.save(model.state_dict(), os.path.join(output_folder, 'model_{:.4f}.pth'.format(best_dice)))
            print("=> saved best model, dice_coef = {:.4f}".format(best_dice))
            trigger = 0

        # early stopping
        if 0 <= config['early_stopping'] <= trigger:
            print("=> early stopping")
            break

        torch.cuda.empty_cache()
