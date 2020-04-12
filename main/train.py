import tensorflow as tf
import numpy as np
from model_graph import Model_graph
from config import cfg
from base import Trainer
import argparse

parser = argparse.ArgumentParser(description='Train')
parser.add_argument('--gpu',default='0',type=str, dest='gpu')

def main():
    global args
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    trainer = Trainer(Model_graph(), cfg)
    trainer.train()

if __name__ == '__main__':
    main()
