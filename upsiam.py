import torch
import torch.nn as nn


class UpBlock(nn.Module):
    def __init__(self, channels, dim_size, X):
        super(UpBlock, self).__init__()
        self.X = X
        self.conv = nn.Conv2d(channels, channels, 1)
        # 损失函数
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.conv.parameters())

    def forward(self, Y):
        self.optim(Y)
        return self.conv(Y) # Y * self.M

    def optim(self, Y):
        def closure():
            z = self.conv(Y) #  * self.M
            loss = self.loss(z, self.X) # + self.ratio * torch.abs(torch.mean(self.conv.weight)) # + torch.abs(torch.mean(self.M))
            loss.backward(retain_graph=True)
            print("Loss {}".format(loss.item()))
            return loss, z

        for i in range(20):
            # print("Epoch {}".format(i))
            self.optimizer.zero_grad()
            self.optimizer.step(closure)# , M_inv=None)

class Mem():
    def __init__(self, amount):
        self.amount = amount
        self.dset = []
        
    def insert_sampler(self, sampler):
        if len(self.dset) <= self.amount:
            self.dset.__delitem__[1]
        self.dset.append(sampler)
    
    def fit(self, net, kernels):
        kernels = torch.cat(self.net.mem).permute(0, 2, 3, 1).reshape(-1, 512)
        kernels_mean = sum(kernels)/len(kernels)
        net.pca.fit(kernels.cpu().numpy())
        trans_vec = net.pca.components_[0]

        return trans_vec, kernels_mean