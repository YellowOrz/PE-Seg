config = {}
config['name'] = None
config['epochs'] = 100
config['batch_size'] = 1

config['arch'] = 'DoubleUNet'
config['deep_supervision'] = False
config['input_channels'] = 3
config['num_classes'] = 1
config['input_w'] = 512
config['input_h'] = 512

config['loss'] = 'BCEDiceLoss'

config['dataset'] = '/home/ls/PE_Seg/PE_data/train/'
config['img_ext'] = '.png'
config['mask_ext'] = '.png'

config['optimizer'] = 'SGD'
config['lr'] = 1e-3
config['weight_decay'] = 1e-4
config['momentum'] = 0.9
config['nesterov'] = False

config['scheduler'] = 'CosineAnnealingLR'
config['min_lr'] = 1e-5
config['factor'] = 0.1
config['patience'] = 2
config['milestones'] = '1.2'
config['gamma'] = 2/3
config['early_stopping'] = -1
config['num_workers'] = 0
