#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-11 15:28:41
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-12-21 20:42:35
# FilePath     : /speech-to-text-wavenet/torch_lyuan/dataset.py
# Description  : 
#-------------------------------------------# 

import torch
from torch.utils.data import Dataset
import utils
import random
import json
import os
import numpy as np
# import tensorflow as tf


# def create(filepath, batch_size=1, repeat=False, buffsize=1000):
#   def _parse(record):
#     keys_to_features = {
#       'uid': tf.FixedLenFeature([], tf.string),
#       'audio/data': tf.VarLenFeature(tf.float32),
#       'audio/shape': tf.VarLenFeature(tf.int64),
#       'text': tf.VarLenFeature(tf.int64)
#     }
#     features = tf.parse_single_example(
#       record,
#       features=keys_to_features
#     )
#     audio = features['audio/data'].values
#     shape = features['audio/shape'].values
#     audio = tf.reshape(audio, shape)
#     audio = tf.contrib.layers.dense_to_sparse(audio)
#     text = features['text']
#     return audio, text, shape[0], features['uid']

#   dataset = tf.data.TFRecordDataset(filepath).map(_parse).batch(batch_size=batch_size)
#   loader = torch.utils.data.DataLoader(dataset, batch_size=32)

#   return loader

class VCTK(Dataset):
    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.mode = mode
        assert self.mode in ['train', 'val']
        if not os.path.exists(self.cfg.datalist):
            raise ValueError('datalist must exists, initial datalist is not supported')
        self.train_filenames, self.test_filenames = json.load(open(self.cfg.datalist, 'r', encoding='utf-8'))
        if self.mode == 'train':
            self.max_wave = 520
            self.max_text = 256
        else:
            self.max_wave = 720
            self.max_text = 256

    def __getitem__(self, idx):
        if self.mode =='train':
            filenames = self.train_filenames[idx]
        else:
            filenames = self.test_filenames[idx]
        wave_path = self.cfg.dataset + filenames[0]
        txt_path = self.cfg.dataset + filenames[1]
        try:
            text_tmp = utils.read_txt(txt_path)  # list
            wave_tmp = utils.read_wave(wave_path) # numpy
        except OSError:
            print(txt_path)
            print(wave_path)
            return self.__getitem__(0)
        wave_tmp = torch.from_numpy(wave_tmp)
        wave = torch.zeros([40, self.max_wave]) # 512 may be too short, if error,fix it
        length_wave = wave_tmp.shape[1]
        # print(length_wave)
        wave[:,:length_wave] = wave_tmp
        # print(txt_path)


        while 27 in text_tmp:
            text_tmp.remove(27)

        length_text = len(text_tmp)
        text_tmp = torch.tensor(text_tmp)
        text = torch.zeros([self.max_text]) # 256 may be too short, fix it, if error
        text[:length_text] = text_tmp
        name = filenames[0].split('/')[-1]

        if length_text >= length_wave:
            sample = {'name':name, 'wave':torch.zeros([40, self.max_wave],dtype=torch.float), 'text':torch.zeros([self.max_text],dtype=torch.float),
                    'length_wave':self.max_wave, 'length_text':self.max_text}
        else:
            sample = {'name':name, 'wave':wave, 'text':text,
                    'length_wave':length_wave, 'length_text':length_text}
        return sample


    def __len__(self):
        if self.mode == 'train':
            return len(self.train_filenames)
        else:
            return len(self.test_filenames)




if __name__ == '__main__':
    # train_filenames, test_filenames = json.load(open('/lyuan/code/speech-to-text-wavenet/data/list.json', 'r', encoding='utf-8'))
    # print(len(train_filenames), train_filenames) #[['/VCTK-Corpus/wav48/p376/p376_076.wav', '/VCTK-Corpus/txt/p376/p376_076.txt'], ['/VCTK-Corpus/wav48/p376/p376_021.wav', '/VCTK-Corpus/txt/p376/p376_021.txt']]
    import config_train as cfg
    # vctk = VCTK(cfg, 'train')
    # length = len(vctk)
    # max_length = 0
    # for i in range(length):
    #     tmp = vctk[i]['wave'].shape[1]
    #     if tmp>max_length:
    #         max_length = tmp
    # print(f'train set {max_length}')
    vctk = VCTK(cfg, 'val')
    length = len(vctk)
    max_wave = 0
    max_text = 0
    for i in range(length):
        length_wave = vctk[i]['length_wave']
        if length_wave > max_wave:
            max_wave = length_wave
        length_text = vctk[i]['length_text']
        if length_text > max_text:
            max_text = length_text
        print(f'val set {i}， {length}, {max_wave}, {max_text}, {length_wave}, {length_text}')



