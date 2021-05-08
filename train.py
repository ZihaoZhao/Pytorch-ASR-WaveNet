#----------------description----------------# 
# Author       : Lei yuan
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-10 17:40:40
# LastEditors  : Zihao Zhao
# LastEditTime : 2021-05-08 12:32:21
# FilePath     : /pytorch-asr-wavenet/train.py
# Description  : 0.001 0-5, 0.0001
#-------------------------------------------# 

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.rnn as  rnn_utils
import deepdish as dd

import config_train as cfg
from dataset import VCTK
import dataset
from wavenet import WaveNet
from sparsity import *
import utils
import visualize as vis

from ctcdecode import CTCBeamDecoder

from tensorboardX import SummaryWriter
import os
import numpy as np

import time
import argparse
from write_excel import *
import torch.onnx

def parse_args():
    '''
    Parse input arguments
    '''
    parser = argparse.ArgumentParser(description='WaveNet for speech recognition.')
    parser.add_argument('--exp', type=str, help='exp dir', default="default")
    parser.add_argument('--resume', action='store_true', help='resume from exp_name/best.pth', default=False)

    parser.add_argument('--batch_size', type=int, help='1, 16, 32', default=16)
    parser.add_argument('--lr', type=float, help='0.001 for tensorflow', default=0.001)

    parser.add_argument('--load_from', type=str, help='.pth', default="/z")

    parser.add_argument('--skip_exist', action='store_true', help='if exist', default=False)
    parser.add_argument('--save_excel', type=str, help='exp.xls', default="default.xls")

    args = parser.parse_args()
    return args

def train(train_loader, scheduler, model, loss_fn, val_loader, writer=None):
    decoder_vocabulary = utils.Data.decoder_vocabulary
    vocabulary = utils.Data.vocabulary
    decoder = CTCBeamDecoder(
        decoder_vocabulary,
        #"_abcdefghijklmopqrstuvwxyz_",
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=27,
        log_probs_input=True
    )
    train_loss_list = list()
    val_loss_list = list()
    
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f'Training epoch {epoch}')
        _loss = 0.0
        step_cnt = 0
        
        _tp, _pred, _pos = 0, 0, 0
        for data in train_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            if epoch == 0 and step_cnt == 0:
                # print("test3")
                loss_val = validate(val_loader, model, loss_fn)
                writer.add_scalar('val/loss', loss_val, epoch)
                best_loss = loss_val
                not_better_cnt = 0
                torch.save(model.state_dict(), cfg.workdir+'/weights/best.pth')
                print("saved", cfg.workdir+'/weights/best.pth', not_better_cnt)
                val_loss_list.append(float(loss_val))
                model.train()    

            logits = model(wave)
            mask = torch.zeros_like(logits)
            for n in range(len(data['length_wave'])):
                mask[:, :, :data['length_wave'][n]] = 1
            logits *= mask

            logits = logits.permute(2, 0, 1)
            logits = F.log_softmax(logits, dim=2)
            if data['text'].size(0) == cfg.batch_size:
                for i in range(cfg.batch_size):
                    if i == 0:
                        text = data['text'][i][0:data['length_text'][i]].cuda()
                    else:
                        text = torch.cat([text, 
                                    data['text'][i][0: data['length_text'][i]].cuda()])
            else:
                continue

            loss = 0.0
            for i in range(logits.size(1)):
                loss += loss_fn(logits[:data['length_wave'][i], i:i+1, :], data['text'][i][0:data['length_text'][i]].cuda(), data['length_wave'][i], data['length_text'][i])
            loss /= logits.size(1)
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            _loss += loss.data

            step_cnt += 1

        _loss /= len(train_loader)
        writer.add_scalar('train/loss', _loss, epoch)
        train_loss_list.append(float(_loss))
        torch.cuda.empty_cache()

        loss_val = validate(val_loader, model, loss_fn)
        writer.add_scalar('val/loss', loss_val, epoch)
        val_loss_list.append(float(loss_val))

        model.train()

        if loss_val < best_loss:
            not_better_cnt = 0
            torch.save(model.state_dict(), cfg.workdir+f'/weights/best.pth')
            print("saved", cfg.workdir+f'/weights/best.pth', not_better_cnt)
            best_loss = loss_val
        else:
            not_better_cnt += 1

        if not_better_cnt > 3:
            write_excel(os.path.join(cfg.work_root, cfg.save_excel), 
                            cfg.exp_name, train_loss_list, val_loss_list)
            # exit()

def validate(val_loader, model, loss_fn):    
    decoder_vocabulary = utils.Data.decoder_vocabulary
    vocabulary = utils.Data.vocabulary
    decoder = CTCBeamDecoder(
        decoder_vocabulary,
        #"_abcdefghijklmopqrstuvwxyz_",
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=27,
        log_probs_input=True
    )
    model.eval()
    _loss = 0.0
    step_cnt = 0
    _tp, _pred, _pos = 0, 0, 0
    with torch.no_grad():
        for data in val_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            logits = model(wave)
            logits = logits.permute(2, 0, 1)
            logits = F.log_softmax(logits + 1e-10, dim=2)
            if data['text'].size(0) == cfg.batch_size:
                for i in range(cfg.batch_size):
                    if i == 0:
                        text = data['text'][i][0:data['length_text'][i]].cuda()
                        # print(data['text'].size())
                        # print(data['length_text'][i])
                    else:
                        text = torch.cat([text, 
                                    data['text'][i][0: data['length_text'][i]].cuda()])
            else:
                continue
            loss = 0.0
            for i in range(logits.size(1)):
                loss += loss_fn(logits[:data['length_wave'][i], i:i+1, :], data['text'][i][0:data['length_text'][i]].cuda(), data['length_wave'][i], data['length_text'][i])
            loss /= logits.size(1)
            _loss += loss.data
            # beam_results, beam_scores, timesteps, out_lens = decoder.decode(logits.permute(1, 0, 2))

            # voc = np.tile(vocabulary, (cfg.batch_size, 1))
            # pred = np.take(voc, beam_results[0][0][:out_lens[0][0]].data.numpy())
            # text_np = np.take(voc, data['text'][0][0:data['length_text'][0]].cpu().numpy().astype(int))

            # tp, pred, pos = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))
            # _tp += tp
            # _pred += pred
            # _pos += pos
            # f1 = 2 * _tp / (_pred + _pos + 1e-10)
            
            step_cnt += 1
    print("Val step", step_cnt, "/", len(val_loader),
            ", loss: ", round(float(_loss/len(val_loader)), 5))

    return _loss/len(val_loader)


def test_acc(val_loader, model, loss_fn):    
    decoder_vocabulary = utils.Data.decoder_vocabulary
    vocabulary = utils.Data.vocabulary
    decoder = CTCBeamDecoder(
        decoder_vocabulary,
        #"_abcdefghijklmopqrstuvwxyz_",
        model_path=None,
        alpha=0,
        beta=0,
        cutoff_top_n=40,
        cutoff_prob=1.0,
        beam_width=100,
        num_processes=4,
        blank_id=27,
        log_probs_input=True
    )
    model.eval()
    _loss = 0.0
    step_cnt = 0
    tps, preds, poses = 0, 0, 0
    f_cnt = 0
    with torch.no_grad():
        for data in val_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            if 1:
                print(data['wave'].size())
                np.savetxt("/zhzhao/dataset/VCTK/c_model_input_txt/"+str(f_cnt)+".txt", data['wave'].flatten())
                print(f_cnt)
                f_cnt += 1
            logits = model(wave)
            logits = logits.permute(2, 0, 1)
            logits = F.log_softmax(logits, dim=2)
            if data['text'].size(0) == cfg.batch_size:
                for i in range(cfg.batch_size):
                    if i == 0:
                        text = data['text'][i][0:data['length_text'][i]].cuda()
                    else:
                        text = torch.cat([text, 
                                    data['text'][i][0: data['length_text'][i]].cuda()])
            else:
                continue
            loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
            _loss += loss.data
            for i in range(logits.size(1)):
                logit = logits[:data['length_wave'][i], i:i+1, :]
                beam_results, beam_scores, timesteps, out_lens = decoder.decode(logit.permute(1, 0, 2))
                voc = np.tile(vocabulary, (cfg.batch_size, 1))
                pred = np.take(voc, beam_results[0][0][:out_lens[0][0]].data.numpy())
                text_np = np.take(voc, data['text'][i][0:data['length_text'][i]].cpu().numpy().astype(int))
                pred = [pred]
                text_np = [text_np]

                tp, pred, pos = utils.evalutes(utils.cvt_np2string(pred), utils.cvt_np2string(text_np))
                tps += tp
                preds += pred
                poses += pos
            f1 = 2 * tps / (preds + poses + 1e-10)
            
            step_cnt += 1
            # if cnt % 10 == 0:
    print("Val step", step_cnt, "/", len(val_loader),
            ", loss: ", round(float(_loss.data/step_cnt), 5))
    print("Val tps:", tps, ",preds:", preds, ",poses:", poses, ",f1:", f1)

    return f1, _loss/len(val_loader), tps, preds, poses

def check_and_mkdir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        
def main():
    args = parse_args()
    cfg.resume      = args.resume
    cfg.exp_name    = args.exp
    cfg.work_root   = '/zhzhao/code/wavenet_torch/torch_lyuan/exp_result/'
    cfg.workdir     = cfg.work_root + args.exp + '/debug'
    cfg.sparse_mode = args.sparse_mode
    cfg.batch_size  = args.batch_size
    cfg.lr          = args.lr
    cfg.load_from   = args.load_from
    cfg.save_excel   = args.save_excel        

    weights_dir = os.path.join(cfg.workdir, 'weights')
    check_and_mkdir(weights_dir)

    print('initial training...')
    print(f'work_dir:{cfg.workdir}, \n\
            pretrained: {cfg.load_from},  \n\
            batch_size: {cfg.batch_size}, \n\
            lr        : {cfg.lr},         \n\
            epochs    : {cfg.epochs},     \n\
            sparse    : {cfg.sparse_mode}')
    writer = SummaryWriter(log_dir=cfg.workdir+'/runs')

    # build train data
    vctk_train = VCTK(cfg, 'train')
    train_loader = DataLoader(vctk_train, batch_size=cfg.batch_size, num_workers=4, shuffle=True, pin_memory=True)
    vctk_val = VCTK(cfg, 'val')
    val_loader = DataLoader(vctk_val, batch_size=cfg.batch_size, num_workers=4, shuffle=False, pin_memory=True)

    # build model
    model = WaveNet(num_classes=28, channels_in=40, dilations=[1,2,4,8,16])
    model = nn.DataParallel(model)
    model.cuda()
    model.train()

    # build loss
    loss_fn = nn.CTCLoss(blank=27)

    if cfg.resume and os.path.exists(cfg.workdir + '/weights/best.pth'):
        model.load_state_dict(torch.load(cfg.workdir + '/weights/best.pth'), strict=True)
        print("loading", cfg.workdir + '/weights/best.pth')
        cfg.load_from = cfg.workdir + '/weights/best.pth'

    scheduler = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-4)
    train(train_loader, scheduler, model, loss_fn, val_loader, writer)

if __name__ == '__main__':
    main()
