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
from datasets import Demo
import argparse

parser = argparse.ArgumentParser(description='Demo')
parser.add_argument('--gpu',default='0', type=str, dest='gpu')
parser.add_argument('--epoch', type=str, dest='epoch')
parser.add_argument('--vis', default=True, dest='vis')
parser.add_argument('--demo', default=True, dest='demo')

def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    cfg.set_vis(args.vis)
    cfg.set_demo(args.demo)
    test(int(args.epoch))

def test_net(tester, data):
    test_data = data[:]
    for batch_id in range(0, len(test_data), cfg.test_batch_size):
        start_id = batch_id
        end_id = min(len(test_data), batch_id + cfg.test_batch_size)    
        imgs = []
        for i in range(start_id, end_id):
            img ,crop_info = generate_batch(test_data[i], stage='test')
            imgs.append(img)
        imgs = np.array(imgs)
        heatmap = tester.predict_one([imgs],batch_id)[0]    
       
def test(test_model):
    d = Demo()
    data = d.load_val_data_with_annot()
    tester = Tester(Model_graph(), cfg)
    tester.load_weights(test_model)
    result = test_net(tester, data)

if __name__ == '__main__':
    main()

