import torch.nn
import argparse


def init_weights_xavier(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def init_weights_kaiming(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.ConvTranspose2d):
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.InstanceNorm2d:
        if m.weight is not None:
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)


def log2tensorboard(writer, log, epoch, tag):
    for k,v in log.items():
        if 'img' in k:
            writer.add_image('PE_Seg/{}_{}'.format(tag, k), v, epoch)
        else:
            writer.add_scalar('PE_Seg/{}_{}'.format(tag, k), v, epoch)


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)


def str2bool(v):
    if v.lower() in ['true', '1']:
        return True
    elif v.lower() in ['false', '0']:
        return False
    else:
        return argparse.ArgumentTypeError('Boolean Value Expected')


def count_params(model):
    return (sum(p.numel()) for p in model.parameters() if p.requires_grad)