import os
import cv2
import numpy as np
import torch
import torchvision
import visdom
from codes.update.updatenet import DistanceNetwork

viz = visdom.Visdom()

BIGNUM = 1e22  # e22

class samples_manager():
    def __init__(self, store_amount):
        # postive samples
        self.pos_features = []
        # Distance matrix stores the square of the euclidean distance between each pair of samples. Initialise it to inf
        self.distance_matrix = torch.ones((store_amount, store_amount), dtype=torch.float32) * BIGNUM
        # Kernel matrix, used to update distance matrix
        self.gram_matrix = torch.ones((store_amount, store_amount), dtype=torch.float32) * BIGNUM
        self.sample_weights = torch.zeros((store_amount, 1), dtype=torch.float32)
        self.store_amount = store_amount
        self.measure_d = DistanceNetwork()

        self.support_set_list = []
        self.support_set = None
        self.image_list = []

        self.init_gt = None

    def insert(self, feat, image=None):
        if len(self.support_set_list) < self.store_amount:
            self.support_set_list.append(feat)
            self.support_set = torch.stack(self.support_set_list)
            if image is not None:
                self.image_list.append(image)

        else:
            self.distance_matrix = self.compute_distance_matrix(self.support_set_list)
            self.gram_matrix = self.compute_gram_matrix(self.support_set_list)
            self.update_sample_space_model(feat, image)
            a = torchvision.utils.make_grid(torch.cat(self.image_list), nrow=10)
            viz.image(a, opts={"title":"sample space"}, win="space")

    def insert_gt(self, init_gt):
        self.init_gt = init_gt

    def update_sample_space_model(self, feat, image=None):
        dist_vector = self.calc_distvector(feat, self.support_set.squeeze())
        new_sample_id = -1
        # Find sample closest to the new sample
        new_sample_min_dist, closest_sample_to_new_sample = torch.min(dist_vector), torch.argmin(dist_vector)
        # Find the closest pair amongst existing samples
        existing_samples_min_dist, closest_existing_sample_pair = torch.min(
            self.distance_matrix.view(-1)), torch.argmin(self.distance_matrix.view(-1))

        closest_existing_sample1, closest_existing_sample2 = self.ind2sub(self.distance_matrix.shape,
                                                                     closest_existing_sample_pair)

        if torch.equal(closest_existing_sample1, closest_existing_sample2):
            os.error('Score matrix diagonal filled wrongly')

        if new_sample_min_dist < existing_samples_min_dist:
            new_sample_id = closest_sample_to_new_sample

            # Update distance matrix and the gram matrix
            self.update_distance_matrix(dist_vector, new_sample_id)

        if new_sample_id >= 0:
            # 第一帧监督
            # will_be_replaced = self.support_set[new_sample_id.item()]
            # old_sim = self.measure_d(self.init_gt, will_be_replaced)
            # new_sim = self.measure_d(self.init_gt, feat)
            #
            # if new_sim > old_sim:
            self.support_set_list[new_sample_id.item()] = feat
            if image is not None:
                self.image_list[new_sample_id.item()] = image

    def update_distance_matrix(self, dist_vector, new_id):
        if new_id >= 0:
            # Update distance matrix
            if self.distance_matrix[:, new_id].shape == dist_vector.t().shape:
                self.distance_matrix[:, new_id] = dist_vector.t()
                self.distance_matrix[new_id, :] = dist_vector
                self.distance_matrix[new_id, new_id] = BIGNUM
            else:
                self.distance_matrix[:, new_id] = dist_vector
                self.distance_matrix[new_id, :] = dist_vector
                self.distance_matrix[new_id, new_id] = BIGNUM
        elif new_id < 0:
            pass
            # The new sample is discared

    def compute_distance_matrix(self, x):
        x = torch.stack(x).squeeze()
        m, n = x.shape
        G = x.mm(x.t())
        H = G.diagonal().repeat((m, 1))
        distance_matrix = H + H.t() - 2 * G
        distance_matrix.fill_diagonal_(BIGNUM)

        return distance_matrix


    def compute_gram_matrix(self, x):
        x = torch.stack(x).squeeze()
        gram_matrix = x.mm(x.t())
        return gram_matrix


    def sub2ind(self, array_shape, rows, cols):
        ind = rows * array_shape[1] + cols
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1]] = -1
        return ind


    def ind2sub(self, array_shape, ind):
        ind[ind < 0] = -1
        ind[ind >= array_shape[0] * array_shape[1]] = -1
        rows = (ind.int() / array_shape[1])
        cols = ind.int() % array_shape[1]
        return rows, cols


    def calc_distvector(self, A, B):
        m = A.shape[0]
        n = B.shape[0]
        M = A.mm(B.t())
        H = A.pow(2).sum(dim=1).repeat(1, n)
        K = B.pow(2).sum(dim=1).repeat(m, 1)

        dist_vec = torch.sqrt(-2 * M + H + K)
        # 第一帧模板进行打分重要性加权
        # similarities = self.measure_d(self.support_set, self.init_gt)
        # # 归一化
        # min = similarities.min()
        # max = similarities.max()
        # similarities = (similarities - min) / (max - min)
        #
        # weighted_dist_vec = dist_vec / similarities
        # return weighted_dist_vec
        return dist_vec