import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import config_train as cfg
from dataset import VCTK
from wavenet import WaveNet
import utils

from tensorboardX import SummaryWriter
import os
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1)
    return parser.parse_args()

def train(args, train_loader, scheduler, model, loss_fn, val_loader, writer=None):
    weights_dir = os.path.join(cfg.workdir, 'weights')
    if not os.path.exists(weights_dir):
        os.mkdir(weights_dir)

    model.train()
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f'training epoch{epoch}')
        _loss = 0.0
        cnt = 0
        for data in train_loader:
            wave = data['wave'].cuda()  # [1, 128, 109]
            text = data['text'].cuda()
            logits = model(wave)
            logits = logits.permute(2, 0, 1)
            loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
            scheduler.zero_grad()
            loss.backward()
            scheduler.step()
            _loss += loss.data
            cnt += 1
            if args.local_rank==0 and cnt % 1000 == 0:
                print("Epoch", epoch,
                        ", train step", cnt, "/", len(train_loader),
                        ", loss: ", round(float(_loss.data/cnt), 5))
        _loss /= len(train_loader)

        with torch.no_grad():
            loss_val = validate(args, val_loader, model, loss_fn)

        if args.local_rank == 0:
            reduce_train_loss = reduce_tensor(_loss.data)
            reduce_val_loss = reduce_tensor(loss_val.data)
            writer.add_scalar('train/loss', reduce_train_loss, epoch)
            writer.add_scalar('val/loss', reduce_val_loss, epoch)

            if loss_val < best_loss:
                torch.save(model.state_dict(), cfg.workdir+'/weights/best.pth')
                best_loss = loss_val


def validate(args, val_loader, model, loss_fn):
    model.eval()
    _loss = 0.0
    cnt = 0
    for data in val_loader:
        wave = data['wave'].cuda()  # [1, 128, 109]
        logits = model(wave)
        logits = logits.permute(2, 0, 1)
        text = data['text'].cuda()
        loss = loss_fn(logits, text, data['length_wave'], data['length_text'])
        _loss += loss.data
        cnt += 1
        if args.local_rank==0 and cnt % 500 == 0:
            print("Val step", cnt, "/", len(val_loader),
                    ", loss: ", round(float(_loss.data/cnt), 5))
    return _loss/len(val_loader)

def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt)
    rt /= dist.get_world_size()
    return rt


def main():
    print('initial training...')
    print(f'work_dir:{cfg.workdir}, pretrained:{cfg.load_from}, batch_size:{cfg.batch_size} lr:{cfg.lr}, epochs:{cfg.epochs}')
    args = parse_args()
    writer = SummaryWriter(log_dir=cfg.workdir+'/runs')

    # distributed training setting
    assert cfg.distributed
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )

    # build dataloader
    vctk_train = VCTK(cfg, 'train')
    train_sample = torch.utils.data.distributed.DistributedSampler(vctk_train, shuffle=True, )
    # train_loader = DataLoader(vctk_train,batch_size=cfg.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    train_loader = DataLoader(vctk_train, batch_size=cfg.batch_size, sampler=train_sample,
                              num_workers=8, pin_memory=True)

    vctk_val = VCTK(cfg, 'val')
    val_sample = torch.utils.data.distributed.DistributedSampler(vctk_val, shuffle=False, )
    # val_loader = DataLoader(vctk_val, batch_size=cfg.batch_size, num_workers=8, shuffle=False, pin_memory=True)
    val_loader = DataLoader(vctk_val, batch_size=cfg.batch_size, sampler=val_sample, num_workers=8,
                              pin_memory=True)

    # build model
    model = WaveNet(num_classes=28, channels_in=20).cuda()
    model = DDP(model, device_ids=[args.local_rank], broadcast_buffers=False)
    # model = nn.DataParallel(model)

    # build loss
    loss_fn = nn.CTCLoss()

    #
    scheduler = optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-4)
    # scheduler = optim.lr_scheduler.MultiStepLR(train_step, milestones=[50, 150, 250], gamma=0.5)

    # train
    train(args, train_loader, scheduler, model, loss_fn, val_loader, writer)
    # val
    # loss = validate(val_loader, scheduler, model, loss_fn)

if __name__ == '__main__':
    main()
