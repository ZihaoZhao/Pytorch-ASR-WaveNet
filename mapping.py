#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-12-20 11:52:22
# LastEditors  : Zihao Zhao
# LastEditTime : 2021-05-08 12:37:45
# FilePath     : /pytorch-asr-wavenet/mapping.py
# Description  : 
#-------------------------------------------# 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as  rnn_utils
# import deepdish as dd

# import config_train as cfg
# from dataset import VCTK
# import dataset
# from wavenet import WaveNet
# from sparsity import *
# import utils
# import visualize as vis

# from ctcdecode import CTCBeamDecoder

# from tensorboardX import SummaryWriter
import os
import numpy as np

# import time
# import argparse
# from write_excel import *

model_pth = "/Users/zzh/Nutstore Files/Server-Code/DLA-explorers/DLA-mapper/data/model/wavenet/wavenet_dense.pth"
pattern_dir = "/Users/zzh/Nutstore Files/Server-Code/DLA-explorers/DLA-c-model/tests/data/wavenet/pattern"
save_dir = "/Users/zzh/Nutstore Files/Server-Code/DLA-explorers/DLA-c-model/tests/data/wavenet/weights"

def main():
    print("Mapping...")
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_weight_txt(model_pth, save_dir)

def save_weight_txt(model_pth, folder):

    name_list = list()
    para_list = list()

    model_weights = torch.load(model_pth, map_location=torch.device('cpu'))

    for name, raw_w in model_weights.items():
        # pytorch OC, IC, K
        # C model K, IC, OC        
        raw_w_save = np.array(raw_w)
        if name.split(".")[-2] != "bn" \
            and name.split(".")[-2] != "bn2" \
            and name.split(".")[-2] != "bn3" \
            and name.split(".")[-1] != "bias":
            # print(name)
            # print(raw_w_save.shape)
            raw_w_save = raw_w_save.transpose((2, 1, 0))
        print(os.path.join(folder, name + '.txt'))
        np.savetxt(os.path.join(folder, name + '.txt'), raw_w_save.flatten())
        
def read_txt():
    layer = "/module.resnet_block_0.0.conv_filter.dilation_conv1d.weight.txt"
    pattern_txt = pattern_dir + layer
    weight_txt  = save_dir + layer
    pattern = np.loadtxt(pattern_txt).reshape((16, 8, 8))
    weight = np.loadtxt(weight_txt).reshape((7, 128, 128))
    # weight = np.loadtxt(weight_txt).reshape((1, 40, 128))
    for i in range(0, 16):
        for j in range(0, 8):
            for k in range(0, 8):
                print(pattern[i, j, k], end = ' ')
            print(" ")
        print(i)
    print("w")
    for j in range(0, 8):
        for k in range(0, 8):
            print(weight[0, j, k], end = ' ')
        print(" ")

if __name__ == "__main__":
    main()
    # read_txt()