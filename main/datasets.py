import os
import numpy as np
import json
import sys

class BP4D(object):    
    dataset_path = '/data0/BP4D/'
    num_AU_points = 10 #5*2
    AUs = ['AU06_1', 'AU06_2', 'AU10_1', 'AU10_2', 'AU12_1', 'AU12_2','AU14_1', 'AU14_2', 'AU17_1', 'AU17_2']
    train_annot_path = os.path.join(dataset_path, 'train_BP4D.json')
    val_annot_path = os.path.join(dataset_path, 'val_BP4D.json')

    def load_train_data(self, score=False):
        with open(self.train_annot_path,'r') as load_f:
            train_data = json.load(load_f)
        return train_data
    
    def load_val_data_with_annot(self):
        with open(self.val_annot_path,'r') as load_f:
            val_data = json.load(load_f)
        return val_data

    def evaluation(self, result, output_dir, db_set):
        result_path = os.path.join(output_dir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f)

class DISFA(object):    
    dataset_path = '/data0/DISFA/'
    num_AU_points = 24 #12*2
    AUs = ['AU01_1', 'AU01_2', 'AU02_1', 'AU02_2','AU04_1', 'AU05_1', 'AU05_2','AU06_1', 'AU06_2',\
    'AU09_1', 'AU09_2','AU09_3','AU12_1','AU12_2','AU15_1','AU15_2','AU17_1','AU17_2',\
    'AU20_1','AU20_1','AU25_1','AU25_2','AU26_1','AU26_2']
    train_annot_path = os.path.join(dataset_path, 'train_DISFA.json')
    val_annot_path = os.path.join(dataset_path, 'test_DISFA.json')

    def load_train_data(self, score=False):
        with open(self.train_annot_path,'r') as load_f:
            train_data = json.load(load_f)
        return train_data
    
    def load_val_data_with_annot(self):
        with open(self.val_annot_path,'r') as load_f:
            val_data = json.load(load_f)
        return val_data

    def evaluation(self, result, output_dir, db_set):
        result_path = os.path.join(output_dir, 'result.json')
        with open(result_path, 'w') as f:
            json.dump(result, f)

