#! -*- coding: utf-8 -*-
# Author: Yahui Liu
# Email: yahui.liu@unitn.it

import os, cv2
import glob
import numpy as np
import scipy.io as sio

def mat2img(mat_path):
    assert os.path.isfile(mat_path)
    return sio.loadmat(mat_path)['groundtruth']

def img_save(img, save_path):
    flags = [cv2.IMWRITE_JPEG_QUALITY, 100]
    cv2.imwrite(save_path, img, flags)

if __name__ == '__main__':
    lab_dir = './annotations/pixel-level'
    save_path = './annotations/pixel-level-im'
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    label_list = glob.glob(os.path.join(lab_dir, '*.mat'))
    for ll in label_list:
        lab = mat2img(ll).astype('uint8')
        print(np.max(lab))
        fname = ll.split('/')[-1].replace('.mat', '.jpg')
        img_save(lab, os.path.join(save_path, fname))
