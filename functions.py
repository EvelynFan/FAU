import os
import numpy as np
import cv2
from config import cfg
import random
import time
import math
cur_dir = os.path.dirname(os.path.abspath(__file__))

def get_lr(epoch):
    for e in cfg.lr_dec_epoch:
        if epoch < e:
            break
    if epoch < cfg.lr_dec_epoch[-1]:
        i = cfg.lr_dec_epoch.index(e)
        return cfg.lr / (cfg.lr_dec_factor ** i)
    else:
        return cfg.lr / (cfg.lr_dec_factor ** len(cfg.lr_dec_epoch))

def normalize_input(img):
    return img - cfg.pixel_means
def denormalize_input(img):
    return img + cfg.pixel_means

def generate_batch(d, stage='train'):
    img = cv2.imread(d['imgpath'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if cfg.demo:
        path_new = cur_dir+d['imgpath']
        img = cv2.imread(path_new, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if img is None:
        print('cannot read ' + d['imgpath'])
        assert 0

    bbox = np.array([0, 0, 256,256])#for test
    x, y, w, h = bbox
    center = np.array([x + w * 0.5, y + h * 0.5])
    scale = np.array([w,h])
    rotation = 0

    if stage == 'train':
        AUs = np.array(d['AUs']).reshape(cfg.num_AU_points, 3).astype(np.float32)        
        AU_coord = AUs[:,:2]
        AU_intensity = AUs[:,2]
        return [normalize_input(img),AU_coord,AU_intensity]
    else:
        crop_info = np.asarray([center[0]-scale[0]*0.5, center[1]-scale[1]*0.5, center[0]+scale[0]*0.5, center[1]+scale[1]*0.5])
        return [normalize_input(img), crop_info]


