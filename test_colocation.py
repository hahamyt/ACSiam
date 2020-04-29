import torch
from siamrpn import *


if __name__ == "__main__":
    # setup GPU device if available
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    # setup model
    net = SiamRPN()
    if net_path is not None:
        net.load_state_dict(torch.load(
            net_path, map_location=lambda storage, loc: storage))
    net = net.to(device)