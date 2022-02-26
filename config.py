config = {}
config['name'] = None
config['epochs'] = 300
config['batch_size'] = 5

config['arch'] = 'DoubleUNet'
config['deep_supervision'] = False
config['input_channels'] = 3
config['num_classes'] = 1
config['input_w'] = 384
config['input_h'] = 512

config['loss'] = 'BCEDiceLoss'

config['dataset'] = '/home/liushuo/dataset/ISIC-2018'
config['img_ext'] = '.png'
config['mask_ext'] = '.png'

config['optimizer'] = 'Adam'
config['lr'] = 1e-5
config['weight_decay'] = 1e-3
config['momentum'] = 0.9
config['nesterov'] = False

config['scheduler'] = 'ReduceLROnPlateau'
config['min_lr'] = 1e-8
config['factor'] = 0.1
config['patience'] = 20
config['milestones'] = '1.2'
config['gamma'] = 2 / 3
config['early_stopping'] = -1
config['num_workers'] = 6   # 并行读取数据
