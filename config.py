import os
import sys
import numpy as np
from datasets import BP4D,DISFA
cur_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(cur_dir, 'lib'))

class Config:
    dataset = 'BP4D' 
    if dataset == 'BP4D':
        dataset = BP4D()
    else:
        dataset = DISFA()     

    data_dir = dataset.dataset_path
    num_AU_points = dataset.num_AU_points
    output_dir = os.path.join(data_dir, 'output')
    model_dir = os.path.join(output_dir, 'models')
    vis_dir = os.path.join(output_dir, 'vis')
    log_dir = os.path.join(output_dir, 'logs')
    backbone = 'resnet50' 
    init_model = '/data0/resnet_v1_50.ckpt'

    input_shape = (256,256)
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    sigma = 2
    pixel_means = np.array([[[123.68, 116.78, 103.94]]])
    #learning rate setting
    lr_dec_epoch = [5, 10]
    lr = 5e-4 
    lr_dec_factor = 10
    end_epoch = 10
    optimizer = 'adam'
    weight_decay = 1e-5
    bn_train = True
    batch_size = 16#16
    test_batch_size = 1

    multi_thread_enable = True
    num_thread = 10
    display = 1
    def set_vis(self, vis=False):
        self.vis = vis
    def set_demo(self, demo=False):
        self.demo = demo

cfg = Config()

if not os.path.exists(cfg.model_dir):
    os.makedirs(cfg.model_dir)
if not os.path.exists(cfg.vis_dir):
    os.makedirs(cfg.vis_dir)
if not os.path.exists(cfg.log_dir):
    os.makedirs(cfg.log_dir)

       
