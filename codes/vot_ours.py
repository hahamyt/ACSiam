#!/usr/bin/python
import vot
from vot import Rectangle
import sys
import os
import cv2  # imread
import torch
import numpy as np
from os.path import realpath, dirname, join
# from memory_profiler import profile
from net import SiamRPNBIG
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect, load_net_weight

# # load net
# net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')

# print(net_file)


# net = SiamRPNBIG().cpu()
# net.load_state_dict(torch.load(net_file, map_location=torch.device('cpu')))
# net.eval().cpu()# .cuda()

# warm up
# for i in range(10):#10
#     net.temple(torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127)).cpu())#.cuda())
#     net(torch.autograd.Variable(torch.FloatTensor(1, 3, 255, 255)), torch.autograd.Variable(torch.FloatTensor(1, 3, 127, 127).cpu()))#.cuda())

# start to track
handle = vot.VOT("rectangle")
Polygon = handle.region()

cx, cy, w, h = Polygon.x, Polygon.y, Polygon.width, Polygon.height # get_axis_aligned_bbox(Polygon)
print(Polygon)
image_file = handle.frame()
if not image_file:
    sys.exit(0)

target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
base_path = "/media/x/_dde_data/workspace/vot-python/"
im = cv2.imread(os.path.join(base_path, image_file))  # HxWxC

# load net
net_file = join(realpath(dirname(__file__)), 'SiamRPNBIG.model')
net = SiamRPNBIG().cpu()
# net.load_state_dict(torch.load(net_file, map_location=torch.device('cpu')))
load_net_weight(net, torch.load(net_file, map_location=torch.device('cpu')))
net.eval()#.cpu()# .cuda()

state = SiamRPN_init(im, target_pos, target_sz, net)  # init tracker
while True:
    image_file = handle.frame()
    if not image_file:
        break
    print(os.path.join(base_path, image_file))
    im = cv2.imread(os.path.join(base_path, image_file))  # HxWxC
    state = SiamRPN_track(state, im)  # track
    res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])

    handle.report(Rectangle(res[0], res[1], res[2], res[3]))

