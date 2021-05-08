#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2021-05-08 12:16:12
# FilePath     : /pytorch-asr-wavenet/config_train.py
# Description  : 
#-------------------------------------------# 
user = 'zzh'
work_root = '/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/'
mode = None # set at dataloader, must be None
resume = True

exp_name = 'dense_32'
dataset = '/zhzhao/dataset/VCTK'
datalist = '/zhzhao/code/wavenet_torch/data/list.json'

batch_size = 32 # reconmendate 32
load_from = ''
epochs = 1000
lr = 1e-2
