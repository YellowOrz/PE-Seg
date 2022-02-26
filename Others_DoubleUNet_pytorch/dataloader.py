import os
import cv2
import random
import numpy as np
import matplotlib.pyplot as plt

root = os.path.join('./PE_data/train')
img_dir = os.path.join(root, "images")
mask_dir = os.path.join(root, "masks")

img_ids = list(sorted(os.listdir(img_dir)))

random.seed(42)

train_img_ids = img_ids[: 392]
val_img_ids = img_ids[392:-1]


# print(len(train_img_ids), len(val_img_ids)) 392, 100
# def preprocess_mask(mask):
#     mask = mask.astype(np.float32)
#     mask[mask == 2.0] = 0.0
#     mask[(mask == 1.0) | (mask == 3.0)] = 1.0
#     return mask


def display_img_grid(img_ids, img_dir, mask_dir, predicted_masks=None):
    cols = 3 if predicted_masks else 2
    rows = len(img_ids)
    figure, ax = plt.subplots(nrows=rows, ncols=cols)
    for i, img_id in enumerate(img_ids):
        img = cv2.imread(os.path.join(img_dir, img_id))
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(os.path.join(mask_dir, img_id.replace(".jpg", ".png")),
                          cv2.IMREAD_UNCHANGED, )
        # mask = preprocess_mask(mask)