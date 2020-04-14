import torch
import torch.nn as nn
import torch.nn.functional as F

class UpBlock(nn.Module):
    def __init__(self, channels, dim_size, X):
        super(UpBlock, self).__init__()
        # setup GPU device if available
        self.cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')

        self.X = X
        self.conv = nn.Conv2d(channels, channels, 1).to(self.device)

        # 损失函数
        self.loss = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.conv.parameters())
        self.mem = Mem(6)

    def forward(self, Y):
        self.optim(Y)

        self.mem.insert(self.conv.weight)
        output = F.conv2d(Y, self.mem.weights[0])
        for weight in self.mem.weights[1:]:
            output += F.conv2d(Y, weight)

        return output/len(self.mem.weights) # Y * self.M

    def optim(self, Y):
        with torch.set_grad_enabled(True):
            def closure():
                z = self.conv(Y) #  * self.M
                loss = self.loss(z, self.X) # + self.ratio * torch.abs(torch.mean(self.conv.weight)) # + torch.abs(torch.mean(self.M))
                loss.backward(retain_graph=True)
                # print("Loss {}".format(loss.item()))
                return loss, z

            for i in range(1):
                # print("Epoch {}".format(i))
                self.optimizer.zero_grad()
                self.optimizer.step(closure)# , M_inv=None)

class Mem():
    def __init__(self, amount):
        self.amount = amount
        self.nets = []
        self.dist = []
        self.M = torch.zeros(amount, amount)
        self.weights = []

        self.first_time = True

    def insert(self, weight):
        # 计算权重的直方图分布
        hist = torch.histc(weight, bins=30)
        if len(self.nets) >= self.amount:
            # TODO
            if self.first_time:
                for i in range(self.amount):
                    for j in range(self.amount):
                        if i == j:
                            self.M[j, i] = 1e10    
                            continue
                        self.M[j, i] = -F.kl_div(self.nets[j], self.nets[i], size_average=None, reduce=None, reduction='batchmean')
                
                self.first_time = False
            else:
                d = torch.zeros(self.amount, 1)
                for i in range(self.amount):
                    d[i] = -F.kl_div(hist, self.nets[i], size_average=None, reduce=None, reduction='batchmean')
                dmin, didx = torch.min(d), torch.argmin(d)
                mmin, midx = torch.min(self.M), self.ind2sub(torch.argmin(self.M))
                if dmin > mmin:
                    self.nets[didx] = hist
                    self.weights[didx] = weight
                    self.M[midx[0], midx[1]] = dmin
                    self.M[midx[1], midx[0]] = dmin
        else:
            # 保存权重
            self.weights.append(weight)
            self.nets.append(hist)
    
    def ind2sub(self, ind):
        ind[ind < 0] = -1
        ind[ind >= self.amount*self.amount] = -1
        rows = (ind.int() / self.amount)
        cols = ind.int() % self.amount
        return rows, cols
