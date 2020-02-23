import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable

from codes.update.memory import ConvLSTM


class MatchingNetwork(nn.Module):
    def __init__(self, input_size=6, batch_size=3, input_dim=256, hidden_dim=[256], kernel_size=(3, 3),
                 learning_rate=1e-3, num_layers=1, batch_first=True, bias=True):
        """
        This is our main network
        :param keep_prob: dropout rate
        :param batch_size:
        :param num_channels:
        :param learning_rate:
        :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
        :param num_classes_per_set:
        :param num_samples_per_class:
        :param image_size:
        """
        super(MatchingNetwork, self).__init__()
        self.use_cuda = False
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.lstm = ConvLSTM(input_size=(input_size, input_size), input_dim=input_dim, hidden_dim=hidden_dim,
                             kernel_size=kernel_size, num_layers=num_layers, batch_first=batch_first, bias=bias,
                             return_all_layers=False)
        self.hidden = None

    def forward(self, support_set_images):
        """
        Main process of the network
        :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
               用于训练LSTM的样本
        :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
               训练样本的标签,这里是热力图
        :param search_region: shape[batch_size,num_channels,image_size,image_size]
               搜索区域
        :param target:
               搜索区域对应的标签
        :return:
        """
        output, self.hidden = self.lstm(support_set_images, self.hidden)  # , self.hidden)
        return output

    def train_net(self, support_set_images, gt_temple):
        output = self.forward(support_set_images)
        for i in range(output[0].size(1)):
            output[0][:, i, :, :, :] = output[0][:, i, :, :, :] + gt_temple
        return output