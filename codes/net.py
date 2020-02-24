# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from codes.run_SiamRPN import tracker_eval
from codes.update.hessianfree import HessianFree
from codes.update.memory import Memory, ConvLSTM
from codes.update.updatenet import MatchingNetwork
import torch
from memory_profiler import profile # 内存占用分析插件

class SiamRPN(nn.Module):
    def __init__(self, size=2, feature_out=512, anchor=5):
        configs = [3, 96, 256, 384, 384, 256]
        configs = list(map(lambda x: 3 if x==3 else x*size, configs))
        feat_in = configs[-1]
        super(SiamRPN, self).__init__()

        self.featureExtract = nn.Sequential(
            nn.Conv2d(configs[0], configs[1] , kernel_size=11, stride=2),
            nn.BatchNorm2d(configs[1]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[1], configs[2], kernel_size=5),
            nn.BatchNorm2d(configs[2]),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[2], configs[3], kernel_size=3),
            nn.BatchNorm2d(configs[3]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[3], configs[4], kernel_size=3),
            nn.BatchNorm2d(configs[4]),
            nn.ReLU(inplace=True),
            nn.Conv2d(configs[4], configs[5], kernel_size=3),
            nn.BatchNorm2d(configs[5]),
        )

        # 用于推理的更新网络
        self.update = nn.Sequential(nn.Conv2d(feat_in, feature_out, 1),
                                    torch.nn.Tanh(),
                                    nn.Conv2d(feature_out, feature_out, 1),)# MatchingNetwork()
        self.update_loss = torch.nn.MSELoss()
        # self.update_optimizer = torch.optim.SGD(self.update.parameters(), lr = 0.01, momentum=0.9)
        self.update_optimizer = HessianFree(self.update.parameters(),
                                            use_gnm=True, verbose=False)
        self.anchor = anchor
        self.feature_out = feature_out

        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        # 原来的算法是,在第一帧直接计算一次kernel,现在我们引入一个LSTM网络,利用存储在Memory中的时序训练样本
        # 推理出kernel
        # 1, 因此先定义一个Memory组件: amount 表示的是存储时序的数目,这里取值为3
        self.memory = Memory(amount=5)
        # 2, 定义embedded的集合
        self.r1_kernel = []
        # 3, 边框回归的组件与原来保持一致,这里不做变化
        self.cls1_kernel = []

        self.cfg = {}

    def forward(self, x):
        x_f = self.featureExtract(x)
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
               F.conv2d(self.conv_cls2(x_f), self.cls1_kernel)

    def featextract(self, x):
        x_f = self.featureExtract(x)
        return x_f

    def kernel(self, z_f):
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)


    def temple(self, z):
        z_f = self.featureExtract(z)
        # 将第一帧的模板保存起来
        self.memory.templete(z_f)
        # 初始化滤波器,包括边框回归的和跟踪打分的
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)

    # @profile(precision=4, stream=open('memory_profiler.log', 'w+'))
    def update_kernel(self):
        gts = self.memory.search_target
        z_f = self.featureExtract(gts.squeeze(0))
        # Update Part
        # 计算,将当前的这些gt样本的语义输出,与template模板的语义输出尽可能靠近
        # minimize the distance between new generated z_f and gt
        current_gt = z_f[-1, :, :, :].unsqueeze(0).repeat((z_f.size(0) - 1, 1, 1, 1))
        z_f = self.update(z_f[:-1, :, :, :])
        # init_gt = self.memory.init_templete.repeat((z_f.size(0), 1, 1, 1))
        # Hessian Free version
        #@profile(precision=4, stream=open('memory_profiler.log', 'w+'))
        def closure():
            z = self.update(z_f)
            loss = self.update_loss(z, current_gt)
            loss.backward(retain_graph=True)# (create_graph=True)
            # print(loss.item())
            return loss, z

        for i in range(1):
            # print("Epoch {}".format(i))
            self.update_optimizer.zero_grad()
            self.update_optimizer.step(closure, M_inv=None)


        ### Normal Version
        # loss = self.update_loss(z_f, init_gt)
        #
        # # optimize process
        # self.update_optimizer.zero_grad()
        # loss.backward()
        # self.update_optimizer.step()


        cls1_kernel_raw = self.conv_cls1(z_f[-1, :, :, :].unsqueeze(0))
        kernel_size = cls1_kernel_raw.data.size()[-1]
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)

        # 用于边框回归的 :为了防止对边框回归过于频繁, 我们在更新的过程中,不更新边框回归的核
        # cls1_kernel_raw = self.conv_cls1(z_f)
        # self.cls1_kernel = cls1_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)

class SiamRPNBIG(SiamRPN):
    def __init__(self):
        super(SiamRPNBIG, self).__init__(size=2)
        self.cfg = {'lr':0.295, 'window_influence': 0.42, 'penalty_k': 0.055, 'instance_size': 271, 'adaptive': True} # 0.383


class SiamRPNvot(SiamRPN):
    def __init__(self):
        super(SiamRPNvot, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr':0.45, 'window_influence': 0.44, 'penalty_k': 0.04, 'instance_size': 271, 'adaptive': False} # 0.355


class SiamRPNotb(SiamRPN):
    def __init__(self):
        super(SiamRPNotb, self).__init__(size=1, feature_out=256)
        self.cfg = {'lr': 0.30, 'window_influence': 0.40, 'penalty_k': 0.22, 'instance_size': 271, 'adaptive': False} # 0.655


if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    import torch

    writer = SummaryWriter()
    model = SiamRPN()
    dummy_input = torch.rand(1, 3, 271, 271)
    with SummaryWriter(comment="Net") as w:
        w.add_graph(model, (dummy_input,))

    writer.close()