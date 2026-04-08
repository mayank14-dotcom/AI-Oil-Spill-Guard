import torch
from model import UNet

m = UNet()
x = torch.randn(1,3,256,256)
print(m(x).shape)
