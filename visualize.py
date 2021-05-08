#----------------description----------------# 
# Author       : Zihao Zhao
# E-mail       : zhzhao18@fudan.edu.cn
# Company      : Fudan University
# Date         : 2020-10-19 16:38:18
# LastEditors  : Zihao Zhao
# LastEditTime : 2020-10-23 15:41:35
# FilePath     : /speech-to-text-wavenet/torch_lyuan/visualize.py
# Description  : 
#-------------------------------------------# 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import config_train as cfg


def visualize(input):
    plt.matshow(input, cmap='hot')
    plt.colorbar()
    plt.show()

def save_visualized_mask(input, tensor_name):
    mask_dir = os.path.join(cfg.vis_dir, "mask")
    if not os.path.exists(mask_dir):
        os.mkdir(mask_dir)
    # plt.matshow(input, cmap='hot')
    for k in range(input.size(2)):
        plt.matshow(input[:,:,k].cpu().numpy(), cmap='hot', vmin = 0, vmax = 1)
        plt.savefig(os.path.join(mask_dir, tensor_name+"_"+str(k)+".png"), dpi=300)
        
def save_visualized_pattern(patterns):
    patterns_dir = os.path.join(cfg.vis_dir, "pattern")
    if not os.path.exists(patterns_dir):
        os.mkdir(patterns_dir)

    for i in range(len(patterns)):
        # print(patterns[i].cpu().numpy())
        plt.matshow(np.frombuffer(patterns[i], dtype=np.float32).reshape(16,16), cmap='hot', vmin = 0, vmax = 1)
        plt.savefig(os.path.join(patterns_dir, str(i)+".png"), dpi=300)


if __name__ == '__main__':
    model = WaveNet(num_classes=28, channels_in=20, dilations=[1,2,4,8,16])
    model = nn.DataParallel(model)
    model.cuda()
    model.load_state_dict(torch.load(cfg.workdir + '/weights/best.pth'))
    save_pattern(model)