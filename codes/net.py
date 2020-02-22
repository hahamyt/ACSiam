# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from codes.run_SiamRPN import tracker_eval
from codes.update.memory import Memory, ConvLSTM
from codes.update.updatenet import MatchingNetwork
import torch


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
        self.update = MatchingNetwork()
        self.update_loss = torch.nn.MSELoss()
        self.update_optimizer = torch.optim.SGD(self.update.lstm.parameters(), lr = 0.01, momentum=0.9)
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
        self.memory = Memory(amount=20)
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
        self.gt_temple = z_f
        # 初始化滤波器,包括边框回归的和跟踪打分的
        r1_kernel_raw = self.conv_r1(z_f)
        cls1_kernel_raw = self.conv_cls1(z_f)
        kernel_size = r1_kernel_raw.data.size()[-1]
        self.r1_kernel = r1_kernel_raw.view(self.anchor*4, self.feature_out, kernel_size, kernel_size)
        self.cls1_kernel = cls1_kernel_raw.view(self.anchor*2, self.feature_out, kernel_size, kernel_size)

    def update_kernel(self, z, state):
        z_f = self.featureExtract(z)
        # Update Part
        # 添加当前的训练样本及其标签
        self.memory.insert_seqs(z_f)
        # 计算标签,　并将其加入到训练样本中
        self.memory.insert_seqs_y(state)

        if len(self.memory.support_seqs_list) >= self.memory.store_amount:
            # update 利用Matching Net的思路, 返回整合后的结果
            # 这里的z_f表示的是序列中不同时刻的输出, 数量与store_amount保持一致
            lstm_outputs = self.update.train_net(self.memory.support_seqs.requires_grad_(True),
                                                 self.gt_temple.requires_grad_(True))
            # 遍历序列, 计算每一个的响应
            all_loss = 0
            for i in range(lstm_outputs[0].size(1)):
                # 每次更新都要更新滤波器, 滤波器的更新需要的等到LSTM更新过后才可执行, 或者等到memory满了才可执行
                r1_kernel_raw = self.conv_r1(lstm_outputs[0][:, i, :, :, :])
                kernel_size = r1_kernel_raw.data.size()[-1]
                self.r1_kernel = r1_kernel_raw.view(self.anchor * 4, self.feature_out, kernel_size, kernel_size)
                step_label = self.calc_step_label(i, state)
                all_loss += self.update_loss(step_label.unsqueeze(0), self.memory.search_target[i, :])

            # optimize process
            self.update_optimizer.zero_grad()
            all_loss.backward()
            self.update_optimizer.step()

            # 用于边框回归的 :为了防止对边框回归过于频繁, 我们在更新的过程中,不更新边框回归的核
            # cls1_kernel_raw = self.conv_cls1(z_f)
            # self.cls1_kernel = cls1_kernel_raw.view(self.anchor * 2, self.feature_out, kernel_size, kernel_size)

    # 计算每一层输出的结果计算得到的位置与尺寸
    def calc_step_label(self, i, state):
        # 计算每一层LSTM输出得到的kernel, 与search_region得到的响应结果
        delta, score = self.forward(self.memory.search_region[:, i, :, :, :].requires_grad_(True))

        p = state['p']
        target_sz = state['target_sz']
        window = state['window']
        target_pos = state['target_pos']

        wc_z = target_sz[1] + p.context_amount * sum(target_sz)
        hc_z = target_sz[0] + p.context_amount * sum(target_sz)
        s_z = np.sqrt(wc_z * hc_z)
        scale_z = p.exemplar_size / s_z
        #
        # d_search = (p.instance_size - p.exemplar_size) / 2
        # pad = d_search / scale_z
        # s_x = s_z + 2 * pad

        delta = delta.permute(1, 2, 3, 0).contiguous().view(4, -1).data.cpu().numpy()
        score = F.softmax(score.permute(1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1, :].cpu().numpy()

        delta[0, :] = delta[0, :] * p.anchor[:, 2] + p.anchor[:, 0]
        delta[1, :] = delta[1, :] * p.anchor[:, 3] + p.anchor[:, 1]
        delta[2, :] = np.exp(delta[2, :]) * p.anchor[:, 2]
        delta[3, :] = np.exp(delta[3, :]) * p.anchor[:, 3]

        def change(r):
            return np.maximum(r, 1. / r)

        def sz(w, h):
            pad = (w + h) * 0.5
            sz2 = (w + pad) * (h + pad)
            return np.sqrt(sz2)

        def sz_wh(wh):
            pad = (wh[0] + wh[1]) * 0.5
            sz2 = (wh[0] + pad) * (wh[1] + pad)
            return np.sqrt(sz2)

        # size penalty
        s_c = change(sz(delta[2, :], delta[3, :]) / (sz_wh(target_sz)))  # scale penalty
        r_c = change((target_sz[0] / target_sz[1]) / (delta[2, :] / delta[3, :]))  # ratio penalty

        penalty = np.exp(-(r_c * s_c - 1.) * p.penalty_k)
        pscore = penalty * score

        # window float
        pscore = pscore * (1 - p.window_influence) + window * p.window_influence
        best_pscore_id = np.argmax(pscore)

        target = delta[:, best_pscore_id] / scale_z
        target_sz = target_sz / scale_z
        lr = penalty[best_pscore_id] * score[best_pscore_id] * p.lr

        res_x = target[0] + target_pos[0]
        res_y = target[1] + target_pos[1]

        res_w = target_sz[0] * (1 - lr) + target[2] * lr
        res_h = target_sz[1] * (1 - lr) + target[3] * lr

        target_pos = np.array([res_x, res_y])
        target_sz = np.array([res_w, res_h])
        return torch.from_numpy(np.concatenate((target_pos, target_sz))).requires_grad_(True)

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