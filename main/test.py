import os
import numpy as np
from config import cfg
import cv2
import sys
import time
import json
from PIL import Image
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import math
import tensorflow as tf
from base import Tester
from model_graph import Model_graph
from functions import generate_batch
from datasets import DISFA,BP4D
import argparse

parser = argparse.ArgumentParser(description='Test')
parser.add_argument('--gpu',default='0', type=str, dest='gpu')
parser.add_argument('--epoch', type=str, dest='epoch')

def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    test(int(args.epoch))

def test_net(tester, data):
    dump_results = []
    start_time = time.time()
    test_data = data[:]
    AUs_result = np.zeros((len(test_data), cfg.num_AU_points, 3))
    area_save = np.zeros(len(test_data))

    for batch_id in range(0, len(test_data), cfg.test_batch_size):
        start_id = batch_id
        end_id = min(len(test_data), batch_id + cfg.test_batch_size)    
        imgs = []
        crop_infos = []
        for i in range(start_id, end_id):
            img ,crop_info = generate_batch(test_data[i], stage='test')
            imgs.append(img)
            crop_infos.append(crop_info)
        imgs = np.array(imgs)
        crop_infos = np.array(crop_infos)
 
        heatmap = tester.predict_one([imgs],batch_id)[0]    
        
        for image_id in range(start_id, end_id):
            for j in range(cfg.num_AU_points):
                hm_j = heatmap[image_id - start_id, :, :, j]
                idx = hm_j.argmax()
                y, x = np.unravel_index(idx, hm_j.shape)
                px = int(math.floor(x + 0.5))
                py = int(math.floor(y + 0.5))
                if 1 < px < cfg.output_shape[1]-1 and 1 < py < cfg.output_shape[0]-1:
                    diff = np.array([hm_j[py][px+1] - hm_j[py][px-1],
                                     hm_j[py+1][px]-hm_j[py-1][px]])
                    diff = np.sign(diff)
                    x += diff[0] * .25
                    y += diff[1] * .25
                AUs_result[image_id, j, :2] = (x * cfg.input_shape[1] / cfg.output_shape[1], y * cfg.input_shape[0] / cfg.output_shape[0])
                AUs_result[image_id, j, 2] = float(hm_j.max()) / 255.0 
            # map back to original images
            for j in range(cfg.num_AU_points):
                AUs_result[image_id, j, 0] = AUs_result[image_id, j, 0] / cfg.input_shape[1] * (\
                crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) + crop_infos[image_id - start_id][0]
                AUs_result[image_id, j, 1] = AUs_result[image_id, j, 1] / cfg.input_shape[0] * (\
                crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1]) + crop_infos[image_id - start_id][1]         
            area_save[image_id] = (crop_infos[image_id - start_id][2] - crop_infos[image_id - start_id][0]) * (crop_infos[image_id - start_id][3] - crop_infos[image_id - start_id][1])
        for i in range(len(AUs_result)):
            result = dict(AUs=AUs_result[i].round(3).tolist())     
            dump_results.append(result)

    return dump_results


def test(test_model):
    d = cfg.dataset
    data = d.load_val_data_with_annot()
    tester = Tester(Model_graph(), cfg)
    tester.load_weights(test_model)
    result = test_net(tester, data)
    d.evaluation(result, cfg.output_dir, cfg.testset)

if __name__ == '__main__':
    main()

