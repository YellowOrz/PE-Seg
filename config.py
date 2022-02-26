config = {}
config['gpu'] = '0'
config['name'] = None
config['epochs'] = 300
config['batch_size'] = 4
config['num_workers'] = 2   # 并行读取数据

config['arch'] = 'DoubleUNet'
config['deep_supervision'] = False
config['input_channels'] = 3
config['num_classes'] = 1
config['input_w'] = 384
config['input_h'] = 512
config['color_jitter'] = [0.1, 0.1, 0.1, 0.1]  # brightness, contrast, saturation, hue

config['loss'] = 'BCEDiceLoss'

config['dataset'] = '/home/xzf/Projects/Datasets/PE_data_edited/PAT*'
config['img_label'] = 'original'      # img文件夹名称
config['mask_label'] = 'label'      # mask文件夹名称
config['test_size'] = 5             # 测试集的序列数量，剩下的序列全都给训练集

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
