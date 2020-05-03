from __future__ import absolute_import, division

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import scipy.signal as ss
from collections import namedtuple
from got10k.trackers import Tracker
from sklearn.decomposition import PCA
from upsiam import *
from utils import *
import visdom
viz = visdom.Visdom()
from tensorboardX import SummaryWriter
writer = SummaryWriter()

class SiamRPN(nn.Module):

    def __init__(self, anchor_num=5):
        super(SiamRPN, self).__init__()
        self.anchor_num = anchor_num
        self.feature = nn.Sequential(
            # conv1
            nn.Conv2d(3, 192, 11, 2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv2
            nn.Conv2d(192, 512, 5, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2),
            # conv3
            nn.Conv2d(512, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv4
            nn.Conv2d(768, 768, 3, 1),
            nn.BatchNorm2d(768),
            nn.ReLU(inplace=True),
            # conv5
            nn.Conv2d(768, 512, 3, 1),
            nn.BatchNorm2d(512))
        
        self.conv_reg_z = nn.Conv2d(512, 512 * 4 * anchor_num, 3, 1)
        self.conv_reg_x = nn.Conv2d(512, 512, 3)
        self.conv_cls_z = nn.Conv2d(512, 512 * 2 * anchor_num, 3, 1)
        self.conv_cls_x = nn.Conv2d(512, 512, 3)
        self.adjust_reg = nn.Conv2d(4 * anchor_num, 4 * anchor_num, 1)
        
    def forward(self, z, x):
        return self.inference(x, **self.learn(z))

    def learn(self, z):
        z = self.feature(z)
        kernel_reg = self.conv_reg_z(z)
        kernel_cls = self.conv_cls_z(z)

        k = kernel_reg.size()[-1]
        kernel_reg = kernel_reg.view(4 * self.anchor_num, 512, k, k)
        kernel_cls = kernel_cls.view(2 * self.anchor_num, 512, k, k)
        
        return kernel_reg, kernel_cls

    def inference(self, x, kernel_reg, kernel_cls):
        x = self.feature(x)
        x_reg = self.conv_reg_x(x)
        x_cls = self.conv_cls_x(x)
        
        out_reg = self.adjust_reg(F.conv2d(x_reg, kernel_reg))
        out_cls = F.conv2d(x_cls, kernel_cls)

        return out_reg, out_cls

class TrackerSiamRPN(Tracker):
    def __init__(self, net_path=None, **kargs):
        super(TrackerSiamRPN, self).__init__(
            name='111', is_deterministic=True)
        self.parse_args(**kargs)

        self.amount = 3
        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        # setup model
        self.net = SiamRPN()
        if net_path is not None:
            self.net.load_state_dict(torch.load(
                net_path, map_location=lambda storage, loc: storage))
        self.net = self.net.to(self.device)

        self.pca0 = PCA(n_components=1)
        self.pca1 = PCA(n_components=1)
        self.descriptors0_mean_tensor = None
        self.descriptors1_mean_tensor = None
        self.trans_vec0 = None
        self.trans_vec1 = None

        # self.trans_vec = None
        # self.mean_descriptor = None
        self.selected_layers = (5, 9) # 5, 9, 12, 15
        self.step = 0

    def parse_args(self, **kargs):
        self.cfg = {
            'exemplar_sz': 127,
            'instance_sz': 271,
            'total_stride': 8,
            'context': 0.5,
            'ratios': [0.33, 0.5, 1, 2, 3],
            'scales': [8,],
            'penalty_k': 0.055,
            'window_influence': 0.42,
            'lr': 0.295}

        for key, val in kargs.items():
            self.cfg.update({key: val})
        self.cfg = namedtuple('GenericDict', self.cfg.keys())(**self.cfg)

    def init(self, image, box):
        image = np.asarray(image)
        # convert box to 0-indexed and center based [y, x, h, w]
        box = np.array([
            box[1] - 1 + (box[3] - 1) / 2,
            box[0] - 1 + (box[2] - 1) / 2,
            box[3], box[2]], dtype=np.float32)
        self.center, self.target_sz = box[:2], box[2:]

        # for small target, use larger search region
        if np.prod(self.target_sz) / np.prod(image.shape[:2]) < 0.004:
            self.cfg = self.cfg._replace(instance_sz=287)

        # generate anchors
        self.response_sz = (self.cfg.instance_sz - \
            self.cfg.exemplar_sz) // self.cfg.total_stride + 1
        self.anchors = self._create_anchors(self.response_sz)

        # create hanning window
        self.hann_window = np.outer(
            np.hanning(self.response_sz),
            np.hanning(self.response_sz))
        self.hann_window = np.tile(
            self.hann_window.flatten(),
            len(self.cfg.ratios) * len(self.cfg.scales))

        # exemplar and search sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # exemplar image
        self.avg_color = np.mean(image, axis=(0, 1))
        exemplar_image = self._crop_and_resize(
            image, self.center, self.z_sz,
            self.cfg.exemplar_sz, self.avg_color)

        # classification and regression kernels
        exemplar_image = torch.from_numpy(exemplar_image).to(
            self.device).permute([2, 0, 1]).unsqueeze(0).float()
        with torch.set_grad_enabled(False):
            self.net.eval()
            self.kernel_reg, self.kernel_cls = self.net.learn(exemplar_image)
        # 将分类的kernel分装在AttentionBlock这个类中
        self.transformer = UpBlock(channels=512, dim_size=4, X=self.kernel_cls, amount=self.amount)
        # self.win = self.hamming_window(self.cfg.instance_sz)[:, :, np.newaxis].repeat(3, 2)

    def update(self, image):
        image = np.asarray(image)
        # search image
        instance_image = self._crop_and_resize(
            image, self.center, self.x_sz,
            self.cfg.instance_sz, self.avg_color)
        # instance_image = np.uint8(instance_image * self.win)

        instance_image_np = instance_image.copy()
        # classification and regression outputs
        instance_image = torch.from_numpy(instance_image).to(
            self.device).permute(2, 0, 1).unsqueeze(0).float()

        # 测试实验部分
        # 添加当前的exampler
        # exemplar image
        # self.avg_color = np.mean(image, axis=(0, 1))
        # exemplar_image = self._crop_and_resize(
        #     image, self.center, self.z_sz,
        #     self.cfg.exemplar_sz, self.avg_color)
        # # For Debug
        # origin_image = exemplar_image.copy()
        # exemplar_image = torch.from_numpy(exemplar_image).to(
        #     self.device).permute([2, 0, 1]).unsqueeze(0).float()
        # viz.image(exemplar_image.squeeze(), win="Exemplar image")
        m = np.array([1.])
        # if self.step%15 == 0:
        with torch.set_grad_enabled(False):
            if self.trans_vec0 is not None:
                e_image = instance_image.clone()
                for i, layer in enumerate(self.net.feature):
                    e_image = layer(e_image)
                    if i in self.selected_layers:
                        featmap = e_image[0, :].clone()
                        if i == self.selected_layers[0]:
                            h0, w0 = featmap.shape[1], featmap.shape[2]
                            featmap = featmap.view(self.descriptors0_mean_tensor.shape[0],
                                                   -1).transpose(0, 1)
                            featmap -= self.descriptors0_mean_tensor.repeat(featmap.shape[0], 1)
                            features0 = featmap.cpu().detach().numpy()
                        else:
                            h1, w1 = featmap.shape[1], featmap.shape[2]
                            featmap = featmap.view(self.descriptors1_mean_tensor.shape[0],
                                                   -1).transpose(0, 1)
                            featmap -= self.descriptors1_mean_tensor.repeat(featmap.shape[0], 1)
                            features1 = featmap.cpu().detach().numpy()
                            break
                        del featmap

                P0 = np.dot(self.trans_vec0, features0.transpose()).reshape(h0, w0)
                P1 = np.dot(self.trans_vec1, features1.transpose()).reshape(h1, w1)

                # m0 = cv2.resize(P0, (self.cfg.instance_sz, self.cfg.instance_sz))
                # m1 = cv2.resize(P1, (self.cfg.instance_sz, self.cfg.instance_sz))
                # m = m1 # m0 + m1
                # instance_image = (instance_image * m).type_as(self.kernel_cls)

                mask0 = max_conn_mask(P0, self.cfg.instance_sz, self.cfg.instance_sz) # cv2.resize(P0, (mask_sz, mask_sz))[np.newaxis, :, :] #
                mask1 = max_conn_mask(P1, self.cfg.instance_sz, self.cfg.instance_sz) # cv2.resize(P1, (mask_sz, mask_sz))[np.newaxis, :, :]
                mask = mask0 + mask1
                mask[mask == 1] = 0
                mask[mask == 2] = 1
                # 下面是用于可视化的代码
                # bboxes = get_bboxes(mask)
                # instance_image = (instance_image * mask).type_as(self.kernel_cls)

                mask_3 = np.concatenate(
                    (np.zeros((2, self.cfg.instance_sz, self.cfg.instance_sz), dtype=np.uint16), mask * 255), axis=0)
                # 将原图同mask相加并展示
                mask_3 = np.transpose(mask_3, (1, 2, 0))
                mask_3 = instance_image_np + mask_3
                mask_3[mask_3[:, :, 2] > 254, 2] = 255
                mask_3 = np.array(mask_3, dtype=np.uint8)


                # draw bboxes
                # for (x, y, w, h) in bboxes:
                #     cv2.rectangle(mask_3, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # instance_image = torch.from_numpy(mask_3).to(
                #                     self.device).permute(2, 0, 1).unsqueeze(0).float()
                cv2.imshow("Debug", mask_3)
                cv2.waitKey(1)
                # viz.heatmap(mask.squeeze(), win="mask")
                # viz.heatmap(mask0.squeeze(), win="mask0")
                # viz.heatmap(mask1.squeeze(), win="mask1")
                # mask[mask < 0] = 0
                # mask[mask > 0] = 1
                # mask = mask.repeat(5, 0).reshape(response.shape[-1])
                # 看看不加在response上，而是直接与图片相乘会怎样
                # TODO
            # # 提取特征
            # z_t = self.net.feature[0:9](exemplar_image)
            # # 面临两种选择：保存特征z_t还是保存内核z_t？
            # # 这里选择保留特征
            # self.transformer.mem.insert(z_t.view(z_t.shape[1],
            #                                      z_t.shape[2] * z_t.shape[3]).transpose(0, 1))
            # if self.trans_vec is not None:
            #     # 反推回当前的特征，将这些特征的共性提炼出来（共性就是都有要跟踪的目标）
            #     w, h = z_t.shape[2], z_t.shape[3]
            #     tmp = z_t.view(z_t.shape[1], -1).transpose(0, 1)
            #     tmp -= self.mean_descriptor.repeat(tmp.shape[0], 1)
            #     tmp = tmp.detach().cpu().numpy()
            #     P = np.dot(self.trans_vec, tmp.transpose()).reshape(w, h)
            #
            #     mask = max_conn_mask(P, mask_sz, mask_sz)
            #     # 下面是用于可视化的代码
            #     # bboxes = get_bboxes(mask)
            #     #
            #     # mask_3 = np.concatenate(
            #     #     (np.zeros((2, 127, 127), dtype=np.uint16), mask * 255), axis=0)
            #     # # 将原图同mask相加并展示
            #     # mask_3 = np.transpose(mask_3, (1, 2, 0))
            #     # mask_3 = origin_image + mask_3
            #     # mask_3[mask_3[:, :, 2] > 254, 2] = 255
            #     # mask_3 = np.array(mask_3, dtype=np.uint8)
            #     #
            #     # # draw bboxes
            #     # for (x, y, w, h) in bboxes:
            #     #     cv2.rectangle(mask_3, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #     #
            #     # cv2.imshow("Debug", mask_3)
            #     # cv2.waitKey(1)
            #
            #     mask = mask.repeat(5, 0).reshape(response.shape[-1])
            #     # self.kernel_cls = self.transformer(
            #     #         self.transformer.mem.examplers[-1] *
            #     #         P[:, np.newaxis, :, :].repeat(c, 1))
            #     #     pass
            # # 计算核
            # # z_t = self.net.conv_cls_z(z_t)
            # # 这里先保存内核
            # # k = z_t.size()[-1]
            # # self.transformer.mem.insert(z_t.view(2 * self.net.anchor_num, 512, k, k))


        # 计算样本位置得分以及边框回归向量
        with torch.set_grad_enabled(False):
            self.net.eval()
            out_reg, out_cls = self.net.inference(
                instance_image, self.kernel_reg, self.kernel_cls)
        # offsets
        offsets = out_reg.permute(
            1, 2, 3, 0).contiguous().view(4, -1).cpu().numpy()
        offsets[0] = offsets[0] * self.anchors[:, 2] + self.anchors[:, 0]
        offsets[1] = offsets[1] * self.anchors[:, 3] + self.anchors[:, 1]
        offsets[2] = np.exp(offsets[2]) * self.anchors[:, 2]
        offsets[3] = np.exp(offsets[3]) * self.anchors[:, 3]
        # scale and ratio penalty
        penalty = self._create_penalty(self.target_sz, offsets)
        # response
        response = F.softmax(out_cls.permute(
            1, 2, 3, 0).contiguous().view(2, -1), dim=0).data[1].cpu().numpy()
        response = response * penalty
        response = (1 - self.cfg.window_influence) * response + \
            self.cfg.window_influence * self.hann_window

        # 测试代码：加入mask的结果
        # response = np.multiply(m, response)
        # START:测试实验部分
        # 进行TDD计算
        if self.step % 1 == 0:
            # kernel_cls = self.calc_kernel(exemplar_image)
            self.transformer.mem.insert(exampler=instance_image)  # exemplar_image)
            if len(self.transformer.mem.examplers) == self.amount:  # and self.step%self.amount == 0:
                viz.images(torch.cat(self.transformer.mem.examplers), win="Sample Space")
                self.step += 1
                descriptors0 = None
                descriptors1 = None
                for image in self.transformer.mem.examplers:
                    # 计算最大特征值对应的特征向量
                    for i, layer in enumerate(self.net.feature):
                        image = layer(image)
                        if i in self.selected_layers:
                            output = image[0, :].clone()
                            output = output.view(image.shape[1], output.shape[1] * output.shape[2])
                            output = output.transpose(0, 1)
                            if i == self.selected_layers[0]:
                                descriptors0 = np.vstack(
                                    (np.zeros((1, image.shape[1])), output.cpu().detach().numpy().copy()))
                            else:
                                descriptors1 = np.vstack(
                                    (np.zeros((1, image.shape[1])), output.cpu().detach().numpy().copy()))
                            del output

                descriptors0 = descriptors0[1:]
                descriptors1 = descriptors1[1:]
                # labels = torch.from_numpy(np.array([i for i in range(len(descriptors0))]))
                # # 使用tensorboard可视化
                # writer.add_embedding(
                #     descriptors0,
                #     metadata=labels,
                #     label_img=None,
                #     global_step=self.step)
                # labels = torch.from_numpy(np.array([i for i in range(len(descriptors1))]))
                # writer.add_embedding(
                #     descriptors1,
                #     metadata=labels,
                #     label_img=None,
                #     global_step=self.step+1)
                # writer.close()
                # 计算descriptor均值，并将其降为0
                descriptors0_mean = sum(descriptors0) / len(descriptors0)
                self.descriptors0_mean_tensor = torch.FloatTensor(descriptors0_mean)
                descriptors1_mean = sum(descriptors1) / len(descriptors1)
                self.descriptors1_mean_tensor = torch.FloatTensor(descriptors1_mean)

                self.pca0.fit(descriptors0)
                self.trans_vec0 = self.pca0.components_[0]

                self.pca1.fit(descriptors1)
                self.trans_vec1 = self.pca1.components_[0]
        # END:测试实验部分
        # END: 测试代码
        viz.heatmap(response.reshape(5, out_cls.shape[2], out_cls.shape[2]).mean(0), win="Score")
        # peak location
        best_id = np.argmax(response)
        offset = offsets[:, best_id] * self.z_sz / self.cfg.exemplar_sz

        # update center
        self.center += offset[:2][::-1]
        self.center = np.clip(self.center, 0, image.shape[:2])

        # update scale
        lr = response[best_id] * self.cfg.lr
        self.target_sz = (1 - lr) * self.target_sz + lr * offset[2:][::-1]
        self.target_sz = np.clip(self.target_sz, 10, image.shape[:2])

        # update exemplar and instance sizes
        context = self.cfg.context * np.sum(self.target_sz)
        self.z_sz = np.sqrt(np.prod(self.target_sz + context))
        self.x_sz = self.z_sz * \
            self.cfg.instance_sz / self.cfg.exemplar_sz

        # return 1-indexed and left-top based bounding box
        box = np.array([
            self.center[1] + 1 - (self.target_sz[1] - 1) / 2,
            self.center[0] + 1 - (self.target_sz[0] - 1) / 2,
            self.target_sz[1], self.target_sz[0]])

        self.step += 1
        return box

    def _create_anchors(self, response_sz):
        anchor_num = len(self.cfg.ratios) * len(self.cfg.scales)
        anchors = np.zeros((anchor_num, 4), dtype=np.float32)

        size = self.cfg.total_stride * self.cfg.total_stride
        ind = 0
        for ratio in self.cfg.ratios:
            w = int(np.sqrt(size / ratio))
            h = int(w * ratio)
            for scale in self.cfg.scales:
                anchors[ind, 0] = 0
                anchors[ind, 1] = 0
                anchors[ind, 2] = w * scale
                anchors[ind, 3] = h * scale
                ind += 1
        anchors = np.tile(
            anchors, response_sz * response_sz).reshape((-1, 4))

        begin = -(response_sz // 2) * self.cfg.total_stride
        xs, ys = np.meshgrid(
            begin + self.cfg.total_stride * np.arange(response_sz),
            begin + self.cfg.total_stride * np.arange(response_sz))
        xs = np.tile(xs.flatten(), (anchor_num, 1)).flatten()
        ys = np.tile(ys.flatten(), (anchor_num, 1)).flatten()
        anchors[:, 0] = xs.astype(np.float32)
        anchors[:, 1] = ys.astype(np.float32)

        return anchors

    def _create_penalty(self, target_sz, offsets):
        def padded_size(w, h):
            context = self.cfg.context * (w + h)
            return np.sqrt((w + context) * (h + context))

        def larger_ratio(r):
            return np.maximum(r, 1 / r)
        
        src_sz = padded_size(
            *(target_sz * self.cfg.exemplar_sz / self.z_sz))
        dst_sz = padded_size(offsets[2], offsets[3])
        change_sz = larger_ratio(dst_sz / src_sz)

        src_ratio = target_sz[1] / target_sz[0]
        dst_ratio = offsets[2] / offsets[3]
        change_ratio = larger_ratio(dst_ratio / src_ratio)

        penalty = np.exp(-(change_ratio * change_sz - 1) * \
            self.cfg.penalty_k)

        return penalty

    def _crop_and_resize(self, image, center, size, out_size, pad_color):
        # convert box to corners (0-indexed)
        size = round(size)
        corners = np.concatenate((
            np.round(center - (size - 1) / 2),
            np.round(center - (size - 1) / 2) + size))
        corners = np.round(corners).astype(int)

        # pad image if necessary
        pads = np.concatenate((
            -corners[:2], corners[2:] - image.shape[:2]))
        npad = max(0, int(pads.max()))
        if npad > 0:
            image = cv2.copyMakeBorder(
                image, npad, npad, npad, npad,
                cv2.BORDER_CONSTANT, value=pad_color)

        # crop image patch
        corners = (corners + npad).astype(int)
        patch = image[corners[0]:corners[2], corners[1]:corners[3]]

        # resize to out_size
        patch = cv2.resize(patch, (out_size, out_size))

        return patch

    def calc_kernel(self, exemplar_image):
        z = self.net.feature(exemplar_image)
        kernel_cls = self.net.conv_cls_z(exemplar_image)
        k = kernel_cls.size()[-1]
        kernel_cls = kernel_cls.view(2 * self.net.anchor_num, 512, k, k)
        return kernel_cls

    def hamming_window(self, hm_len):
        # build 2d window
        bw2d = np.outer(ss.hamming(hm_len), np.ones(hm_len))
        bw2d = np.sqrt(bw2d * bw2d.T)
        bw2d = (bw2d.max() - bw2d) / (bw2d.max() - bw2d.min())
        return 1 - bw2d