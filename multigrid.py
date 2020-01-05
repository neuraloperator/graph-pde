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
from timeit import default_timer

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
        h = 1.0 / (np.sqrt(x.size()[1]) - 1.0)

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


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)

        self.conv11 = GCNConv(32, 32)
        self.conv21 = GCNConv(32, 32)
        self.conv31 = GCNConv(32, 32)

        # self.conv12 = GCNConv(32, 32)
        # self.conv22 = GCNConv(32, 32)
        # self.conv32 = GCNConv(32, 32)
        self.fc2 = nn.Linear(32, 1)


    def forward(self, data):
        x, edge_index, mask_index = data.x, data.edge_index, data.mask_index

        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        x = F.relu(x)
        x = self.conv11(x, edge_index)
        x = F.relu(x)
        x = self.conv21(x, edge_index)
        x = F.relu(x)
        x = self.conv31(x, edge_index)

        # x = F.relu(x)   
        # x = self.conv12(x, edge_index)
        # x = F.relu(x)
        # x = self.conv22(x, edge_index)
        # x = F.relu(x)
        # x = self.conv32(x, edge_index)
        x = x[mask_index]
        x = F.relu(x)
        x = self.fc2(x)

        return x


def grid(n_x, n_y):

    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))
    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []
    for y in range(n_y):
        for x in range(n_x):
            i = y * n_x + x
            if(x != n_x-1):
                edge_index.append((i, i + 1))
                edge_attr.append((1, 0, 0))
                edge_index.append((i + 1, i))
                edge_attr.append((-1, 0, 0))

            if(y != n_y-1):
                edge_index.append((i, i+n_x))
                edge_attr.append((0, 1, 0))
                edge_index.append((i+n_x, i))
                edge_attr.append((0, -1, 0))

    X = torch.tensor(grid, dtype=torch.float)
    #Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0,1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr

# s_all = [25, 31, 41, 49, 61, 81, 121]
s=65

rel_err = np.zeros((7,))
abs_err = np.zeros((7,))

for q in range(1):
    # s = s_all[q]

    print(s)


    data_loader = MatReader('/home/nikola/Workspace/Data/DarcyFlow/piececonst_r' + str(s) + '_N1024.mat')
    x_train = data_loader.read_field('coeff').view(1024,s,s)[:,0:s-1,0:s-1]
    y_train = data_loader.read_field('sol').view(1024,s,s)[:,0:s-1,0:s-1]

    data_loader.load_file('/home/nikola/Workspace/Data/DarcyFlow/piececonst_r' + str(s) + '_N10000.mat')
    x_test = data_loader.read_field('coeff').view(10000,s,s)[:,0:s-1,0:s-1]
    y_test = data_loader.read_field('sol').view(10000,s,s)[:,0:s-1,0:s-1]

    s = 64




    n_x = 64
    n_y = 64
    X, edge_index, edge_attr = grid(n_x, n_y)
    print(edge_index.shape, edge_attr.shape)

    depth = 4

    edge_index_global = []
    edge_attr_global = []
    X_global = []
    num_nodes = 0

    # build connected graph
    for l in range(depth):
        h_x_l = n_x // (2 ** l)
        h_y_l = n_y // (2 ** l)
        n_l = h_x_l * h_y_l


        X, edge_index_inner, edge_attr_inner = grid(h_y_l, h_x_l)

        # update index
        edge_index_inner = edge_index_inner + num_nodes
        edge_index_global.append(edge_index_inner)
        edge_attr_global.append(edge_attr_inner)
        # construct X
        X_l = torch.tensor(l, dtype=torch.float).repeat(n_l, 1)
        X = torch.cat([X, X_l], dim=1)
        X_global.append(X)

        index1 = torch.tensor(range(n_l), dtype=torch.long)
        index1 = index1 + num_nodes
        num_nodes += n_l

        # #construct inter-graph edge
        if l != depth-1:
            index2 = np.array(range(n_l//4)).reshape(h_x_l//2, h_y_l//2)  # torch.repeat is different from numpy
            index2 = index2.repeat(2, axis = 0).repeat(2, axis = 1)
            index2 = torch.tensor(index2).reshape(-1)
            index2 = index2 + num_nodes

            edge_index_inter1 = torch.cat([index1,index2], dim=-1).reshape(2,-1)
            edge_index_inter2 = torch.cat([index2,index1], dim=-1).reshape(2,-1)
            edge_index_inter = torch.cat([edge_index_inter1, edge_index_inter2], dim=1)

            edge_attr_inter1 = torch.tensor((0, 0, 1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter2 = torch.tensor((0, 0,-1), dtype=torch.float).repeat(n_l, 1)
            edge_attr_inter = torch.cat([edge_attr_inter1, edge_attr_inter2], dim=0)

            edge_index_global.append(edge_index_inter)
            edge_attr_global.append(edge_attr_inter)



    X = torch.cat(X_global, dim=0)
    edge_index = torch.cat(edge_index_global, dim=1)
    edge_attr = torch.cat(edge_attr_global, dim=0)
    mask_index = torch.tensor(range(n_x * n_y), dtype=torch.long)
    print(X.shape,  edge_index.shape, edge_attr.shape, mask_index.shape)


    a = torch.zeros((num_nodes,1))
    train_data = []
    for j in range(1024):
        a[:s**2,:] = x_train[j,:,:].reshape(-1,1)
        train_data.append(Data(x=torch.cat([X,a], dim=1), y=y_train[j,:].reshape(-1,), edge_index=edge_index, edge_attr=edge_attr, mask_index=mask_index))

    test_data = []
    for j in range(9600):
        a[:s**2,:] = x_test[j,:,:].reshape(-1,1)
        test_data.append(Data(x=torch.cat([X,a], dim=1), y=y_test[j,:].reshape(-1,), edge_index=edge_index, edge_attr=edge_attr, mask_index=mask_index))


    batch_size = 64
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    device = torch.device('cuda')

    loss_func = LpLoss(size_average=False)

    n_out_pixels_train = 1024.0*(s**2)

    model = Net().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10,
                            verbose=True, threshold=0.0001, threshold_mode='rel',
                            cooldown=0, min_lr=0, eps=1e-8)

    for epoch in range(500):
        model.train()
        train_err = 0.0 
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = F.mse_loss(out.view(-1), batch.y.view(-1), reduction='sum')

            loss.backward()
            optimizer.step()

            train_err += loss.item()

        train_err = np.sqrt(train_err/n_out_pixels_train)
        scheduler.step(train_err)

        model.eval()
        test_error_rel = 0.0
        test_error_abs = 0.0

        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                pred = model(batch)
                test_error_rel += loss_func(pred.view(batch_size, s**2), batch.y.view(batch_size, s**2)).item()
                test_error_abs += loss_func.abs(pred.view(batch_size, s**2), batch.y.view(batch_size, s**2)).item()


        test_error_rel /= 9600
        test_error_abs /= 9600

        rel_err[q] = test_error_rel
        abs_err[q] = test_error_abs

        print(epoch, rel_err[q], abs_err[q], train_err)

    # scipy.io.savemat('graph_piecewise.mat', mdict={'rel_err': rel_err, 'abs_err': abs_err})
