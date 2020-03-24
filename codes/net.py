# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from update.hessianfree import HessianFree
from update.memory import Memory, ConvLSTM
import torch
from memory_profiler import profile # 内存占用分析插件
import visdom
import itertools

viz = visdom.Visdom()

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
        self.anchor = anchor
        self.feature_out = feature_out

        self.conv_r1 = nn.Conv2d(feat_in, feature_out*4*anchor, 3)
        self.conv_r2 = nn.Conv2d(feat_in, feature_out, 3)
        self.conv_cls1 = nn.Conv2d(feat_in, feature_out*2*anchor, 3)
        self.conv_cls2 = nn.Conv2d(feat_in, feature_out, 3)
        self.regress_adjust = nn.Conv2d(4*anchor, 4*anchor, 1)

        # 2, 定义embedded的集合
        self.r1_kernel = []
        # 3, 边框回归的组件与原来保持一致,这里不做变化
        self.cls1_kernel = []
        self.cfg = {}
        
    def forward(self, x, exampler):
        """
        :param: exampler表示的是当前帧的exampler
        :param: x表示的是搜索区域
        """
        exampler = self.cls_kernel_calc(exampler)
        x_f = self.featureExtract(x) # 这里的x_f表示的是搜索区域， 我们应该加权的是什么特征？应该是目标的特征吧
        return self.regress_adjust(F.conv2d(self.conv_r2(x_f), self.r1_kernel)), \
               F.conv2d(self.conv_cls2(x_f), self.attention.forward(exampler))

    def cls_kernel_calc(self, Y):
        z_f = self.featureExtract(Y)
        # 初始化滤波器,包括边框回归的和跟踪打分的
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = cls1_kernel_raw.data.size()[-1]
        current_cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)
        return current_cls1_kernel

    def featextract(self, x):
        x_f = self.featureExtract(x)
        return x_f

    def kernel(self, z_f):
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size).requires_grad_(True)

    def temple(self, z):
        z_f = self.featureExtract(z)
        # 初始化滤波器,包括边框回归的和跟踪打分的
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)
        # 将分类的kernel分装在AttentionBlock这个类中
        self.attention = AttentionBlock(batch_size=10, channels=self.feature_out, dim_size=4, X=self.cls1_kernel)

    def weigth_init(self, m):
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            init.constant_(m.bias.data,0.1)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0,0.01)
        
class ChannelSELayer(nn.Module):
    """
    Re-implementation of Squeeze-and-Excitation (SE) block described in:
        *Hu et al., Squeeze-and-Excitation Networks, arXiv:1709.01507*
    """

    def __init__(self, num_channels, reduction_ratio=2):
        """
        :param num_channels: No of input channels
        :param reduction_ratio: By how much should the num_channels should be reduced
        :param base_temple: weighted features according to base_temple
        """
        super(ChannelSELayer, self).__init__()
        num_channels_reduced = num_channels // reduction_ratio
        self.reduction_ratio = reduction_ratio
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_tensor, target_state=None):
        """
        :param input_tensor: X, shape = (batch_size, num_channels, H, W)
        :return: output tensor
        :target_state: try to get into this situation
        """
        pass
        batch_size, num_channels, H, W = input_tensor.size()
        input_tensor = input_tensor.view(batch_size, num_channels, -1)
        target_state = torch.rand((batch_size, num_channels, 16))

        for i in range(batch_size):
            input_tensor[i, :, :].t()*Param
        # # Average along each channel
        # squeeze_tensor = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)

        # # channel excitation
        # fc_out_1 = self.relu(self.fc1(squeeze_tensor))
        # fc_out_2 = self.sigmoid(self.fc2(fc_out_1))

        # a, b = squeeze_tensor.size()
        # output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        # return output_tensor


class AttentionBlock(nn.Module):
    def __init__(self, batch_size, channels, dim_size, X):
        super(AttentionBlock, self).__init__()
        self.X = X
        M = torch.randn((batch_size, channels, dim_size, dim_size), requires_grad=True)
        self.M = torch.nn.Parameter(M)
        self.register_parameter("M",self.M) # 注册参数
        # 自定义正则化项系数
        # punishment_ratio = torch.tensor([1.0], requires_grad=True)
        # self.ratio = torch.nn.Parameter(punishment_ratio)
        # self.register_parameter("ratio",self.ratio) # 注册参数

        self.conv = nn.Conv2d(channels, channels, 1)
        # 损失函数
        self.loss = torch.nn.MSELoss()
        self.optimizer = HessianFree(itertools.chain(self.conv.parameters()), use_gnm=True, verbose=False)

        self.debug_loss = []

    def forward(self, Y):
        self.optim(Y)
        return self.conv(Y) # Y * self.M

    def optim(self, Y):
        def closure():
            z = self.conv(Y) #  * self.M
            loss = self.loss(z, self.X) # + self.ratio * torch.abs(torch.mean(self.conv.weight)) # + torch.abs(torch.mean(self.M))
            loss.backward(create_graph=True)
            # print("Loss {}".format(loss.item()))
            self.debug_loss.append(loss.item())
            return loss, z

        for i in range(10):
            # print("Epoch {}".format(i))
            self.optimizer.zero_grad()
            self.optimizer.step(closure, M_inv=None)
        
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