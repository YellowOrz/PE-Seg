from torchvision import transforms
from tqdm import tqdm
from glob import glob
import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset


class Data(Dataset):

    def __init__(self, img_ids, img_dir, mask_dir, num_classes, transform=None):
        self.img_ids = img_ids
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.num_classes = num_classes
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):

        img_id = self.img_ids[idx]
        img = cv2.imread(os.path.join(self.img_dir, img_id))
        # print(img.shape)
        # img = cv2.imread(os.path.join(self.img_dir, "%d.png" % int(img_id)))
        # print(img)
        # mask = []
        # for i in range(self.num_classes):
        #     mask.append(cv2.imread(os.path.join(self.mask_dir, str(i), img_id + self.mask_ext)))
        #     mask = np.dstack(mask)

        # mask_id = self.mask_ids[idx]
        mask = cv2.imread(os.path.join(self.mask_dir, img_id))

        if len(img.shape)==2:
            img = np.tile(img[..., None], (1, 1, 3))
        if img.shape[2]>3:
            img = img [..., :3]


        # img = np.array(img)
        # mask = np.array(mask)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']

        img = img.astype('float32') / 255
        img = img.transpose(2, 0, 1)
        # print(img.shape)

        mask = mask.astype('float32') / 255
        mask = mask.transpose(2, 0, 1)[0, :, :]

        return img, mask, {'img_ids': img_id}