# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import random
import sys
import os
import numpy as np
from os.path import realpath, dirname, join

from torch.autograd import Variable

from codes.net import SiamRPNvot
from codes.options import opts
from codes.run_SiamRPN import SiamRPN_init, SiamRPN_track
from codes.utils import get_axis_aligned_bbox, cxy_wh_2_rect, load_net_weight, get_subwindow_tracking, overlap_ratio
sys.path.append('../')

# load net
# net = SiamRPNvot()
from codes.net import SiamRPNBIG


if opts['seed'] is not None:
    np.random.seed(opts['seed'])
    random.seed(opts['seed'])
    torch.manual_seed(opts['seed'])
    torch.cuda.manual_seed(opts['seed'])
    torch.cuda.manual_seed_all(opts['seed'])
    torch.backends.cudnn.deterministic = True

net = SiamRPNBIG()
net = load_net_weight(net, torch.load(join(realpath(dirname(__file__)),
                    'SiamRPNBIG.model'), map_location=torch.device('cpu')))

net.eval().cpu()

# image and init box
# image_files = sorted(glob.glob('./bag/*.jpg'))
# init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]

image_files = sorted(glob.glob('/media/x/_dde_data/girl/color/*.jpg'))
init_rbox = [301.46,135.15,338.2,137.57,327.25,304.09,290.51,301.67]
[cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

# 得到ground truth
gts = []
all_ious = []
# gt = np.loadtxt("./bag/groundtruth.txt", delimiter=',')
gt = np.loadtxt("/media/x/_dde_data/girl/groundtruth.txt", delimiter=',')
#将cxcywh转成xywh
for i in range(len(gt)):
    rect = np.array(get_axis_aligned_bbox(gt[i]))
    rect[:2] = rect[:2] - rect[2:] / 2
    gts.append(rect)

# tracker init
target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
im = cv2.imread(image_files[0])  # HxWxC
state = SiamRPN_init(im, target_pos, target_sz, net)

# tracking and visualization
toc = 0
for f, image_file in enumerate(image_files):
    im = cv2.imread(image_file)
    tic = cv2.getTickCount()
    state = SiamRPN_track(state, im)  # track
    toc += cv2.getTickCount()-tic
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
    res = [int(l) for l in res]
    cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
    cv2.imshow('SiamRPN', im)
    cv2.waitKey(1)

    # 传入目标的位置和大小, 还有当前帧的模板
    # if f > net.memory.store_amount and f % 5 == 0:
    #     net.update_kernel()
    iou = overlap_ratio(res, gts[f])
    all_ious.append(iou)

    # if iou < 0.1:
    #     print("跟踪失败")

print('Tracking Speed {:.1f}fps'.format((len(image_files)-1)/(toc/cv2.getTickFrequency())))
print('Mean IOU {:.2f}'.format(np.sum(all_ious)/len(gts)))