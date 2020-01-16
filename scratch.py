import torch

a = torch.ones(4,1)
b = torch.ones(1,4)
print(a.reshape(4,1,1).repeat(2,1,3).shape)
