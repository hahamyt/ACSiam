import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class UpBlock(nn.Module):
    def __init__(self, channels, dim_size, X, amount):
        super(UpBlock, self).__init__()
        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.amount = amount
        self.channels = channels
        self.X = X
        self.conv = nn.Conv2d(channels, channels, 1).to(self.device)

        # 损失函数
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.conv.parameters())
        self.mem = Mem(amount)

    def forward(self, Y):
        self.optim(Y)

        # self.mem.insert(self.conv.weight)
        # output = F.conv2d(Y, self.mem.weights[0])
        # for weight in self.mem.weights[1:]:
        #     output += F.conv2d(Y, weight)

        return self.conv(Y)  # output/len(self.mem.weights) # Y * self.M

    def optim(self, Y):
        with torch.set_grad_enabled(True):
            def closure():
                z = self.conv(Y) #  * self.M
                loss = self.loss(z, self.X) # + self.ratio * torch.abs(torch.mean(self.conv.weight)) # + torch.abs(torch.mean(self.M))
                loss.backward(retain_graph=True)
                print("Loss {}".format(loss.item()))
                return loss, z

            for i in range(10):
                # print("Epoch {}".format(i))
                self.optimizer.zero_grad()
                self.optimizer.step(closure)# , M_inv=None)

class Mem():
    def __init__(self, amount):
        self.amount = amount
        self.examplers = []

    def insert(self, exampler):
        # _, c, w, h = exampler.shape
        # if len(self.examplers) <= self.amount:
        #     exampler[:,:,:,0:int(w/2/2)] = np.random.randint(255)
        #     exampler[:,:,:,w-int(w/2/2):w] = np.random.randint(255)
        #     exampler[:,:,0:int(h/2/2),:] = np.random.randint(255)
        #     exampler[:,:,h-int(h/2/2):h,:] = np.random.randint(255)

        if len(self.examplers) >= self.amount:
            self.examplers.__delitem__(1)
        self.examplers.append(exampler)


