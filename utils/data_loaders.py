import cv2
import json
import numpy as np
import os
import io
import random
from glob import glob
import torch.utils.data.dataset


class DatasetSeq(torch.utils.data.dataset.Dataset):
    def __init__(self, dir, img_label, mask_label, phase, test_size, num_classes, transform=None):
        self.num_classes = num_classes
        self.transform = transform
        assert '*' in dir, '数据集必须为包含很多段图像序列，在填写路径的时候序列名称用*表示'
        dirs = [os.path.join(d, '%s', '*') for d in sorted(glob(dir))]
        if phase == 'train':
            dirs = dirs[:-test_size]
        else:
            dirs = dirs[-test_size:]
        self.img_path, self.mask_path = [], []
        for d in dirs:
            self.img_path += sorted(glob(d % img_label))
            self.mask_path += sorted(glob(d % mask_label))
        assert len(self.img_path) == len(self.mask_path), '图像的数量{}跟label的数量{}不一致'.format(len(self.img_path),
                                                                                         len(self.mask_path))

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img = cv2.imread(self.img_path[idx], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.mask_path[idx], cv2.IMREAD_GRAYSCALE)
        if self.transform is not None:
            img, mask = self.transform(img, mask)
        return img, mask

