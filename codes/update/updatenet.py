import torch.nn as nn
import numpy as np
import torch
import torchvision.models as models

#
# class MatchingNetwork(nn.Module):
#     def __init__(self, input_size=6, batch_size=3, input_dim=256, hidden_dim=[256], kernel_size=(3, 3),
#                  learning_rate=1e-3, num_layers=1, batch_first=True, bias=True):
#         """
#         This is our main network
#         :param keep_prob: dropout rate
#         :param batch_size:
#         :param num_channels:
#         :param learning_rate:
#         :param fce: Flag indicating whether to use full context embeddings(i.e. apply an LSTM on the CNN embeddings)
#         :param num_classes_per_set:
#         :param num_samples_per_class:
#         :param image_size:
#         """
#         super(MatchingNetwork, self).__init__()
#         self.use_cuda = False
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         self.lstm = ConvLSTM(input_size=(input_size, input_size), input_dim=input_dim, hidden_dim=hidden_dim,
#                              kernel_size=kernel_size, num_layers=num_layers, batch_first=batch_first, bias=bias,
#                              return_all_layers=False)
#         self.hidden = None
#
#     def forward(self, support_set_images):
#         """
#         Main process of the network
#         :param support_set_images: shape[batch_size,sequence_length,num_channels,image_size,image_size]
#                用于训练LSTM的样本
#         :param support_set_y_one_hot: shape[batch_size,sequence_length,num_classes_per_set]
#                训练样本的标签,这里是热力图
#         :param search_region: shape[batch_size,num_channels,image_size,image_size]
#                搜索区域
#         :param target:
#                搜索区域对应的标签
#         :return:
#         """
#         output, self.hidden = self.lstm(support_set_images, self.hidden)  # , self.hidden)
#         # 输出最后一次的输出
#         return output[-1][:, -1, :, :].unsqueeze(0)
#
#     def train_net(self, support_set_images, gt_temple):
#         output = self.forward(support_set_images)
#         for i in range(output[0].size(1)):
#             output[0][:, i, :, :, :] = output[0][:, i, :, :, :] + gt_temple
#         return output

class DistanceNetwork(nn.Module):
    """
    This model calculates the cosine distance between each of the support set embeddings and the target image embeddings.
    """

    def __init__(self):
        super(DistanceNetwork, self).__init__()

    def forward(self, support_set, input_image):
        """
        forward implement
        :param support_set:the embeddings of the support set images.shape[sequence_length,batch_size,64]
        :param input_image: the embedding of the target image,shape[batch_size,64]
        :return:shape[batch_size,sequence_length]
        """

        eps = 1e-10
        similarities = []
        for support_image in support_set:
            sum_support = torch.sum(torch.pow(support_image, 2), 1)
            support_manitude = sum_support.clamp(eps, float("inf")).rsqrt()
            dot_product = input_image.unsqueeze(1).bmm(support_image.unsqueeze(2)).squeeze()
            cosine_similarity = dot_product * support_manitude
            similarities.append(cosine_similarity)
        similarities = torch.stack(similarities)
        return similarities.t()


