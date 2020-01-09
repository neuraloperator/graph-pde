import torch
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt
import random

import numpy as np
import scipy.io
import h5py

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x_size.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)


class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape)-1,-1,-1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def  set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

def gen_4stencil_edges(s):

    n = 4*((s-2)**2) + 12*(s-2) + 8

    edges = torch.zeros(2,n,dtype=torch.long)

    boundry_top = range(1,s-2 + 1)
    boundry_bottom = range((s-1)*s + 1, s**2 - 2 + 1)
    boundry_left = [j*s for j in range(1,s-2 + 1)]
    boundry_right = [j*s - 1 for j in range(2, s-1 + 1)]

    count = 0
    for j in range(s**2):
        #Corner points
        if j == 0:
            edges[0,count] = j
            edges[1,count] = 1
            count += 1

            edges[0,count] = j
            edges[1,count] = s
            count += 1
        elif j == s-1:
            edges[0,count] = j
            edges[1,count] = s-2
            count += 1

            edges[0,count] = j
            edges[1,count] = 2*s - 1
            count += 1
        elif j == (s-1)*s:
            edges[0,count] = j
            edges[1,count] = (s-2)*s
            count += 1

            edges[0,count] = j
            edges[1,count] = (s-1)*s + 1
            count += 1
        elif j == s**2 - 1:
            edges[0,count] = j
            edges[1,count] = s**2 - 2
            count += 1

            edges[0,count] = j
            edges[1,count] = (s-1)*s - 1
            count += 1
        #Boundry points
        elif j in boundry_top:
            edges[0,count] = j
            edges[1,count] = j - 1
            count += 1

            edges[0,count] = j
            edges[1,count] = j + 1
            count += 1

            edges[0,count] = j
            edges[1,count] = j + s
            count += 1
        elif j in boundry_left:
            edges[0,count] = j
            edges[1,count] = j - s 
            count += 1

            edges[0,count] = j
            edges[1,count] = j + 1
            count += 1

            edges[0,count] = j
            edges[1,count] = j + s
            count += 1
        elif j in boundry_right:
            edges[0,count] = j
            edges[1,count] = j - s
            count += 1

            edges[0,count] = j
            edges[1,count] = j + s
            count += 1

            edges[0,count] = j
            edges[1,count] = j - 1
            count += 1
        elif j in boundry_bottom:
            edges[0,count] = j
            edges[1,count] = j - 1
            count += 1

            edges[0,count] = j
            edges[1,count] = j + 1
            count += 1

            edges[0,count] = j
            edges[1,count] = j - s
            count += 1
        #Interior points
        else:
            edges[0,count] = j
            edges[1,count] = j - s
            count += 1

            edges[0,count] = j
            edges[1,count] = j - 1
            count += 1

            edges[0,count] = j
            edges[1,count] = j + 1
            count += 1

            edges[0,count] = j
            edges[1,count] = j + s
            count += 1

    return edges


data_loader = MatReader('piececonst_r61_graph_N1024.mat')
x_train = data_loader.read_field('x')
y_train = data_loader.read_field('y')

data_loader.load_file('piececonst_r61_graph_N10000.mat')
x_test = data_loader.read_field('x')
y_test = data_loader.read_field('y')

n_x = 61
n_y = 61

edge_index = []
edge_attr = []
for y in range(n_y):
    for x in range(n_x):
        i = y * n_x + x
        if(x != n_x-1):
            edge_index.append((i, i + 1))
            edge_attr.append((1, 0))
            edge_index.append((i + 1, i))
            edge_attr.append((-1, 0))

        if(y != n_y-1):
            edge_index.append((i, i+n_x))
            edge_attr.append((0, 1))
            edge_index.append((i+n_x, i))
            edge_attr.append((0, -1))

edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
edge_attr = torch.tensor(edge_attr, dtype=torch.float)

dim = 2

train_data = []
for j in range(1024):
    train_data.append(Data(x=x_train[j,:,:].view(61**2,3), y=y_train[j,:].view(-1,), edge_index=edge_index, edge_attr=edge_attr))

test_data = []
for j in range(10000):
    test_data.append(Data(x=x_test[j,:,:].view(61**2,3), y=y_test[j,:].view(-1,), edge_index=edge_index, edge_attr=edge_attr))


train_loader = DataLoader(train_data, batch_size=1, shuffle=True)
test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

class Net_MP(nn.Module):
    def __init__(self):
        super(Net_MP, self).__init__()
        nn1 = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, dim * 32))
        nn1_short = nn.Linear(dim, dim * 32)
        self.conv1 = NNConv(dim, 32, nn1, aggr='add')

        nn2 = nn.Sequential(nn.Linear(dim, 16), nn.ReLU(), nn.Linear(16, 1024))
        nn2_short = nn.Linear(dim, 1024)
        self.conv2 = NNConv(32, 32, nn2, aggr='add')

        self.fc1 = torch.nn.Linear(32, 32)
        self.fc2 = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        print(x.size(), edge_index.size(), edge_attr.size())
        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x = F.relu(self.conv2(x, edge_index, edge_attr))
        #x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GCNConv(3, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv21 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv21(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        return x


device = torch.device('cuda')

loss_func = LpLoss(size_average=False)

model = Net_MP().cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

for epoch in range(1000):
    model.train()
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        loss = F.mse_loss(out.view(-1), batch.y.view(-1))

        loss.backward()
        optimizer.step()

    model.eval()
    test_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_error += loss_func(pred.view(1000, 61**2), batch.y.view(1000, 61**2)).item()

    print(epoch, test_error/10000)
