import torch
import torch.nn as nn
import torch.nn.functional as func

class GANLoss(nn.Module):
    def __init__(self, t1, t2, beta):
        super(GANLoss, self).__init__()
        self.t1 = t1
        self.t2 = t2
        self.beta = beta
        return

    def forward(self, anchor, positive, negative):
        pass