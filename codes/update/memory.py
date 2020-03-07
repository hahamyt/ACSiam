import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2
import Augmentor
from torchvision.transforms import Resize, Compose, transforms
import torchvision.models as models
from update.sample_manger import samples_manager
from utils import SampleGenerator, gen_bboxes, crop_image, overlap_ratio

class Memory():
    def __init__(self, amount):
        self.store_amount = amount
        # 用于存储离散的样本,便于操作
        self.support_set_list = []
        self.support_set = None
        # self.featureExtractor = models.squeezenet1_0(pretrained=True).features
        self.manager = samples_manager(amount)
        self.neg_set = []
        p = Augmentor.Pipeline()
        p.rotate(probability=0.1, max_right_rotation=10, max_left_rotation=0)
        p.random_distortion(probability=0.1, grid_width=2, grid_height=2, magnitude=4)
        p.flip_left_right(probability=0.2)
        p.flip_top_bottom(probability=0.1)
        p.random_erasing(probability=0.01, rectangle_area=0.2)
        p.flip_random(probability=0.1)
        self.transform = transforms.Compose([
            p.torch_transform(),
            transforms.ToTensor(),
        ])

        self.pos_samples = []
        self.neg_samples = []

    def insert_support_gt(self, feat, gt):
        # with torch.no_grad():
        # feat = self.featureExtractor.forward(gt.unsqueeze(0)).view(1, -1)
        self.manager.insert(feat, gt)

        self.support_set = torch.stack(self.manager.support_set_list)
        pass

    def insert_init_gt(self, init_gt):
        # init_gt = self.featureExtractor(init_gt).view(1, -1)
        self.manager.insert_gt(init_gt.view(1, 1, -1))

    def insert_neg_set(self, neg_samples):
        if len(self.neg_set) >= self.store_amount:
            self.neg_set.__delitem__(0)
        self.neg_set.append(neg_samples)

    def sample_boxes(self, im, bbox):
        self.size = (im.shape[1], im.shape[0])
        # for update
        self.pos_generator = SampleGenerator('gaussian', self.size, 0.1, 1.3)
        self.neg_generator = SampleGenerator('uniform', self.size, 2, 1.3)

        # generate bounding box rect
        pos_rects = gen_bboxes(SampleGenerator('gaussian', self.size, 0.1, 1.3),
                               bbox, 200, [0.7, 1])

        overlap_pos = [0.7, 1]
        while len(pos_rects) == 0 and overlap_pos[0] > 0.3:
            overlap_pos[0] -= 0.1
            pos_rects = gen_bboxes(SampleGenerator('gaussian', self.size, 0.1, 1.3),
                                   bbox, 500, overlap_pos)

        neg_rects = np.random.permutation(np.concatenate([
            gen_bboxes(SampleGenerator('uniform', self.size, 1, 1.6),
                       bbox, 2000, [0, 0.3])]))
        neg_rects = np.random.permutation(neg_rects)

        # Extract pos/neg features
        pos_regions = self.extract_regions(im, pos_rects)
        neg_regions = self.extract_regions(im, neg_rects)

        return pos_regions, neg_regions

    # extract frames' regions of every rect
    def extract_regions(self, frame, rects, toTensor=True):
        regions = []
        for r in rects:
            if toTensor:
                regions.append(crop_image(frame, r, img_size=127, padding=0).unsqueeze(0))
            else:
                regions.append(crop_image(frame, r, img_size=127, padding=0).unsqueeze(0))
        regions = torch.cat(regions)
        return regions












class ConvLSTMCell(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_size: (int, int)
            Height and width of input tensor as (height, width).
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        # 将输入和输出混合起来
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        # 将混合后的结果进行卷积计算
        combined_conv = self.conv(combined)
        # 将卷积后的结果分离开来,得到一些中间结果, 其中g等价于公式中的tanh的计算过程
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)
        # 计算下一个状态H和cell output C
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)
        # 返回计算结果
        return h_next, c_next

    def init_hidden(self, batch_size):
        return (Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cpu(),
                Variable(torch.zeros(batch_size, self.hidden_dim, self.height, self.width)).cpu())


class ConvLSTM(nn.Module):

    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.height, self.width = input_size

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_size=(self.height, self.width),
                                          input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is None:
            hidden_state = self._init_hidden(batch_size=input_tensor.size(0))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param