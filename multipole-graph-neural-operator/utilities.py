import torch
import numpy as np
import scipy.io
import h5py
import sklearn.metrics
from torch_geometric.data import Data
import torch.nn as nn
from scipy.ndimage import gaussian_filter

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reading data
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
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if len(self.mean.shape) == len(sample_idx[0].shape):
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if len(self.mean.shape) > len(sample_idx[0].shape):
                std = self.std[:,sample_idx]+ self.eps # T*batch*n
                mean = self.mean[:,sample_idx]

        # x is in shape of batch*n or T*batch*n
        x = (x * std) + mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
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
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x

class DenseNet_sin(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet_sin, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

    def forward(self, x):
        for j, l in enumerate(self.layers):
            x = l(x)
            if j != self.n_layers - 1:
                x = torch.sin(x)

        return x

# generate graphs on square domain
class SquareMeshGenerator(object):
    def __init__(self, real_space, mesh_size):
        super(SquareMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.s = mesh_size[0]

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T
    
    def ball_connectivity(self, r):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid)
        rbf = np.exp(-pwd**2/sigma**2)
        sample = np.random.binomial(1,rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def get_grid(self):
        return torch.tensor(self.grid, dtype=torch.float)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            else:
                edge_attr = np.zeros((self.n_edges, 2*self.d+2))
                edge_attr[:,0:2*self.d] = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
                edge_attr[:, 2 * self.d +1] = theta[self.edge_index[1]]
        else:
            xy = self.grid[self.edge_index.T].reshape((self.n_edges,-1))
            if theta is None:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:])
            else:
                edge_attr = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

    def get_boundary(self):
        s = self.s
        n = self.n
        boundary1 = np.array(range(0, s))
        boundary2 = np.array(range(n - s, n))
        boundary3 = np.array(range(s, n, s))
        boundary4 = np.array(range(2 * s - 1, n, s))
        self.boundary = np.concatenate([boundary1, boundary2, boundary3, boundary4])

    def boundary_connectivity2d(self, stride=1):

        boundary = self.boundary[::stride]
        boundary_size = len(boundary)
        vertice1 = np.array(range(self.n))
        vertice1 = np.repeat(vertice1, boundary_size)
        vertice2 = np.tile(boundary, self.n)
        self.edge_index_boundary = np.stack([vertice2, vertice1], axis=0)
        self.n_edges_boundary = self.edge_index_boundary.shape[1]
        return torch.tensor(self.edge_index_boundary, dtype=torch.long)

    def attributes_boundary(self, f=None, theta=None):
        # if self.edge_index_boundary == None:
        #     self.boundary_connectivity2d()
        if f is None:
            if theta is None:
                edge_attr_boundary = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
            else:
                edge_attr_boundary = np.zeros((self.n_edges_boundary, 2*self.d+2))
                edge_attr_boundary[:,0:2*self.d] = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
                edge_attr_boundary[:, 2 * self.d] = theta[self.edge_index_boundary[0]]
                edge_attr_boundary[:, 2 * self.d +1] = theta[self.edge_index_boundary[1]]
        else:
            xy = self.grid[self.edge_index_boundary.T].reshape((self.n_edges_boundary,-1))
            if theta is None:
                edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:])
            else:
                edge_attr_boundary = f(xy[:,0:self.d], xy[:,self.d:], theta[self.edge_index_boundary[0]], theta[self.edge_index_boundary[1]])

        return torch.tensor(edge_attr_boundary, dtype=torch.float)

# generate graphs with sampling
class RandomMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size, attr_features=1):
        super(RandomMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size
        self.attr_features = attr_features

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
                self.m = self.n

        self.idx = np.array(range(self.n))
        self.grid_sample = self.grid


    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.grid_sample = self.grid[self.idx]
        return self.idx

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float)

    def ball_connectivity(self, r, is_forward=False):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]

        if is_forward:
            print(self.edge_index.shape)
            self.edge_index = self.edge_index[:, self.edge_index[0] >= self.edge_index[1]]
            print(self.edge_index.shape)
            self.n_edges = self.edge_index.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long)

    def torus1d_connectivity(self, r):
        grid = self.grid_sample
        pwd0 = sklearn.metrics.pairwise_distances(grid, grid)

        grid1 = grid
        grid1[:,0] = grid[:,0]+1
        pwd1 = sklearn.metrics.pairwise_distances(grid, grid1)

        PWD = np.stack([pwd0,pwd1], axis=2)
        pwd = np.min(PWD, axis=2)

        self.edge_index = np.vstack(np.where(pwd <= r))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def gaussian_connectivity(self, sigma):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        rbf = np.exp(-pwd**2/sigma**2)
        sample = np.random.binomial(1,rbf)
        self.edge_index = np.vstack(np.where(sample))
        self.n_edges = self.edge_index.shape[1]
        return torch.tensor(self.edge_index, dtype=torch.long)

    def attributes(self, f=None, theta=None):
        if f is None:
            if theta is None:
                edge_attr = self.grid[self.edge_index.T].reshape((self.n_edges, -1))
            else:
                theta = theta[self.idx]
                edge_attr = np.zeros((self.n_edges, 2 * self.d + 2*self.attr_features))
                edge_attr[:, 0:2 * self.d] = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
                edge_attr[:, 2 * self.d : 2 * self.d + self.attr_features] = theta[self.edge_index[0]].view(-1, self.attr_features)
                edge_attr[:, 2 * self.d + self.attr_features: 2 * self.d + 2*self.attr_features] = theta[self.edge_index[1]].view(-1, self.attr_features)
        else:
            xy = self.grid_sample[self.edge_index.T].reshape((self.n_edges, -1))
            if theta is None:
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:])
            else:
                theta = theta[self.idx]
                edge_attr = f(xy[:, 0:self.d], xy[:, self.d:], theta[self.edge_index[0]], theta[self.edge_index[1]])

        return torch.tensor(edge_attr, dtype=torch.float)

# # generate two-level graph
class RandomTwoMeshGenerator(object):
    def __init__(self, real_space, mesh_size, sample_size, induced_point):
        super(RandomTwoMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_size
        self.m_i = induced_point

        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        if self.m > self.n:
                self.m = self.n

        self.idx = np.array(range(self.n))
        self.idx_i = self.idx
        self.idx_both = self.idx
        self.grid_sample = self.grid
        self.grid_sample_i = self.grid
        self.grid_sample_both = self.grid


    def sample(self):
        perm = torch.randperm(self.n)
        self.idx = perm[:self.m]
        self.idx_i = perm[self.m: self.m+self.m_i]
        self.idx_both = perm[: self.m+self.m_i]
        self.grid_sample = self.grid[self.idx]
        self.grid_sample_i = self.grid[self.idx_i]
        self.grid_sample_both = self.grid[self.idx_both]
        return self.idx, self.idx_i, self.idx_both

    def get_grid(self):
        return torch.tensor(self.grid_sample, dtype=torch.float),  \
               torch.tensor(self.grid_sample_i, dtype=torch.float), \
               torch.tensor(self.grid_sample_both, dtype=torch.float)

    def ball_connectivity(self, r11, r12, r22):
        pwd = sklearn.metrics.pairwise_distances(self.grid_sample)
        pwd12 = sklearn.metrics.pairwise_distances(self.grid_sample, self.grid_sample_i)
        pwd22 = sklearn.metrics.pairwise_distances(self.grid_sample_i)
        self.edge_index = np.vstack(np.where(pwd <= r11))
        self.edge_index_12 = np.vstack(np.where(pwd12 <= r12))
        self.edge_index_12[1,:] = self.edge_index_12[1,:] + self.m
        self.edge_index_21 =  self.edge_index_12[[1,0],:]
        self.edge_index_22 = np.vstack(np.where(pwd22 <= r22)) + self.m
        self.n_edges = self.edge_index.shape[1]
        self.n_edges_12 = self.edge_index_12.shape[1]
        self.n_edges_22 = self.edge_index_22.shape[1]

        return torch.tensor(self.edge_index, dtype=torch.long), \
               torch.tensor(self.edge_index_12, dtype=torch.long), \
               torch.tensor(self.edge_index_21, dtype=torch.long), \
               torch.tensor(self.edge_index_22, dtype=torch.long)


    def attributes(self, theta=None):
        if theta is None:
            edge_attr = self.grid_sample_both[self.edge_index.T].reshape((self.n_edges, -1))
            edge_attr_12 = self.grid_sample_both[self.edge_index_12.T].reshape((self.n_edges_12, -1))
            edge_attr_21 = self.grid_sample_both[self.edge_index_21.T].reshape((self.n_edges_12, -1))
            edge_attr_22 = self.grid_sample_both[self.edge_index_22.T].reshape((self.n_edges_22, -1))
        else:
            theta = theta[self.idx_both]
            edge_attr = np.zeros((self.n_edges, 3 * self.d))
            edge_attr[:, 0:2 * self.d] = self.grid_sample_both[self.edge_index.T].reshape((self.n_edges, -1))
            edge_attr[:, 2 * self.d] = theta[self.edge_index[0]]
            edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[1]]
            edge_attr_12 = np.zeros((self.n_edges_12, 3 * self.d))
            edge_attr_12[:, 0:2 * self.d] = self.grid_sample_both[self.edge_index_12.T].reshape((self.n_edges_12, -1))
            edge_attr_12[:, 2 * self.d] = theta[self.edge_index_12[0]]
            edge_attr_12[:, 2 * self.d + 1] = theta[self.edge_index_12[1]]
            edge_attr_21 = np.zeros((self.n_edges_12, 3 * self.d))
            edge_attr_21[:, 0:2 * self.d] = self.grid_sample_both[self.edge_index_21.T].reshape((self.n_edges_12, -1))
            edge_attr_21[:, 2 * self.d] = theta[self.edge_index_21[0]]
            edge_attr_21[:, 2 * self.d + 1] = theta[self.edge_index_21[1]]
            edge_attr_22 = np.zeros((self.n_edges_22, 3 * self.d))
            edge_attr_22[:, 0:2 * self.d] = self.grid_sample_both[self.edge_index_22.T].reshape((self.n_edges_22, -1))
            edge_attr_22[:, 2 * self.d] = theta[self.edge_index_22[0]]
            edge_attr_22[:, 2 * self.d + 1] = theta[self.edge_index_22[1]]

        return torch.tensor(edge_attr, dtype=torch.float), \
               torch.tensor(edge_attr_12, dtype=torch.float), \
               torch.tensor(edge_attr_21, dtype=torch.float), \
               torch.tensor(edge_attr_22, dtype=torch.float)

# generate multi-level graph
class RandomMultiMeshGenerator(object):
    def __init__(self, real_space, mesh_size, level, sample_sizes):
        super(RandomMultiMeshGenerator, self).__init__()

        self.d = len(real_space)
        self.m = sample_sizes
        self.level = level

        assert len(sample_sizes) == level
        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        self.idx = []
        self.idx_all = None
        self.grid_sample = []
        self.grid_sample_all = None
        self.edge_index = []
        self.edge_index_down = []
        self.edge_index_up = []
        self.edge_attr = []
        self.edge_attr_down = []
        self.edge_attr_up = []
        self.n_edges_inner = []
        self.n_edges_inter = []

    def sample(self):
        self.idx = []
        self.grid_sample = []

        perm = torch.randperm(self.n)
        index = 0
        for l in range(self.level):
            self.idx.append(perm[index: index+self.m[l]])
            self.grid_sample.append(self.grid[self.idx[l]])
            index = index+self.m[l]
        self.idx_all = perm[:index]
        self.grid_sample_all = self.grid[self.idx_all]
        return self.idx, self.idx_all

    def get_grid(self):
        grid_out = []
        for grid in self.grid_sample:
            grid_out.append(torch.tensor(grid, dtype=torch.float))
        return grid_out, torch.tensor(self.grid_sample_all, dtype=torch.float)

    def ball_connectivity(self, radius_inner, radius_inter):
        assert len(radius_inner) == self.level
        assert len(radius_inter) == self.level - 1

        self.edge_index = []
        self.edge_index_down = []
        self.edge_index_up = []
        self.n_edges_inner = []
        self.n_edges_inter = []
        edge_index_out = []
        edge_index_down_out = []
        edge_index_up_out = []

        index = 0
        for l in range(self.level):
            pwd = sklearn.metrics.pairwise_distances(self.grid_sample[l])
            edge_index = np.vstack(np.where(pwd <= radius_inner[l])) + index
            self.edge_index.append(edge_index)
            edge_index_out.append(torch.tensor(edge_index, dtype=torch.long))
            self.n_edges_inner.append(edge_index.shape[1])
            index = index + self.grid_sample[l].shape[0]

        index = 0
        for l in range(self.level-1):
            pwd = sklearn.metrics.pairwise_distances(self.grid_sample[l], self.grid_sample[l+1])
            edge_index = np.vstack(np.where(pwd <= radius_inter[l])) + index
            edge_index[1, :] = edge_index[1, :] + self.grid_sample[l].shape[0]
            self.edge_index_down.append(edge_index)
            edge_index_down_out.append(torch.tensor(edge_index, dtype=torch.long))
            self.edge_index_up.append(edge_index[[1,0],:])
            edge_index_up_out.append(torch.tensor(edge_index[[1,0],:], dtype=torch.long))
            self.n_edges_inter.append(edge_index.shape[1])
            index = index + self.grid_sample[l].shape[0]

        edge_index_out = torch.cat(edge_index_out, dim=1)
        edge_index_down_out = torch.cat(edge_index_down_out, dim=1)
        edge_index_up_out = torch.cat(edge_index_up_out, dim=1)

        return edge_index_out, edge_index_down_out, edge_index_up_out

    def get_edge_index_range(self):
        # in order to use graph network's data structure,
        # the edge index shall be stored as tensor instead of list
        # we concatenate the edge index list and label the range of each level

        edge_index_range = torch.zeros((self.level,2), dtype=torch.long)
        edge_index_down_range = torch.zeros((self.level-1,2), dtype=torch.long)
        edge_index_up_range = torch.zeros((self.level-1,2), dtype=torch.long)

        n_edge_index = 0
        for l in range(self.level):
            edge_index_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + self.edge_index[l].shape[1]
            edge_index_range[l, 1] = n_edge_index

        n_edge_index = 0
        for l in range(self.level-1):
            edge_index_down_range[l, 0] = n_edge_index
            edge_index_up_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + self.edge_index_down[l].shape[1]
            edge_index_down_range[l, 1] = n_edge_index
            edge_index_up_range[l, 1] = n_edge_index

        return edge_index_range, edge_index_down_range, edge_index_up_range

    def attributes(self, theta=None):
        self.edge_attr = []
        self.edge_attr_down = []
        self.edge_attr_up = []

        if theta is None:
            for l in range(self.level):
                edge_attr = self.grid_sample_all[self.edge_index[l].T].reshape((self.n_edges_inner[l], 2*self.d))
                self.edge_attr.append(torch.tensor(edge_attr))

            for l in range(self.level - 1):
                edge_attr_down = self.grid_sample_all[self.edge_index_down[l].T].reshape((self.n_edges_inter[l], 2*self.d))
                edge_attr_up = self.grid_sample_all[self.edge_index_up[l].T].reshape((self.n_edges_inter[l], 2*self.d))
                self.edge_attr_down.append(torch.tensor(edge_attr_down))
                self.edge_attr_up.append(torch.tensor(edge_attr_up))
        else:
            theta = theta[self.idx_all]

            for l in range(self.level):
                edge_attr = np.zeros((self.n_edges_inner[l], 2 * self.d + 2))
                edge_attr[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index[l].T].reshape(
                    (self.n_edges_inner[l], 2 * self.d))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[l][0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[l][1]]
                self.edge_attr.append(torch.tensor(edge_attr, dtype=torch.float))

            for l in range(self.level - 1):
                edge_attr_down = np.zeros((self.n_edges_inter[l], 2 * self.d + 2))
                edge_attr_up = np.zeros((self.n_edges_inter[l], 2 * self.d + 2))

                edge_attr_down[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index_down[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_down[:, 2 * self.d] = theta[self.edge_index_down[l][0]]
                edge_attr_down[:, 2 * self.d + 1] = theta[self.edge_index_down[l][1]]
                self.edge_attr_down.append(torch.tensor(edge_attr_down, dtype=torch.float))

                edge_attr_up[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index_up[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_up[:, 2 * self.d] = theta[self.edge_index_up[l][0]]
                edge_attr_up[:, 2 * self.d + 1] = theta[self.edge_index_up[l][1]]
                self.edge_attr_up.append(torch.tensor(edge_attr_up, dtype=torch.float))

        edge_attr_out = torch.cat(self.edge_attr, dim=0)
        edge_attr_down_out = torch.cat(self.edge_attr_down, dim=0)
        edge_attr_up_out = torch.cat(self.edge_attr_up, dim=0)
        return edge_attr_out, edge_attr_down_out, edge_attr_up_out

# generate graph, with split and assemble
class RandomGridSplitter(object):
    def __init__(self, grid, resolution, d=2, m=200, l=1, radius=0.25):
        super(RandomGridSplitter, self).__init__()

        self.grid = grid
        self.resolution = resolution
        self.n = resolution**d
        self.d = d
        self.m = m
        self.l = l
        self.radius = radius

        assert self.n % self.m == 0
        self.num = self.n // self.m # number of sub-grid

    def get_data(self, theta, edge_features=1):

        data = []
        for i in range(self.l):
            perm = torch.randperm(self.n)
            perm = perm.reshape(self.num, self.m)

            for j in range(self.num):
                idx = perm[j,:].reshape(-1,)
                grid_sample = self.grid.reshape(self.n,-1)[idx]
                theta_sample = theta.reshape(self.n,-1)[idx]

                X = torch.cat([grid_sample,theta_sample],dim=1)

                pwd = sklearn.metrics.pairwise_distances(grid_sample)
                edge_index = np.vstack(np.where(pwd <= self.radius))
                n_edges = edge_index.shape[1]
                edge_index = torch.tensor(edge_index, dtype=torch.long)

                if edge_features == 0:
                    edge_attr = grid_sample[edge_index.T].reshape(n_edges, -1)
                else:
                    edge_attr = np.zeros((n_edges, 2*self.d+2))
                    a = theta_sample[:,0]
                    edge_attr[:, :2*self.d] = grid_sample[edge_index.T].reshape(n_edges, -1)
                    edge_attr[:, 2*self.d] = a[edge_index[0]]
                    edge_attr[:, 2*self.d+1] = a[edge_index[1]]
                    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1, cuda=False):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.num * self.l // batch_size2

        out = torch.zeros(self.n, )
        if cuda:
            out = out.cuda()
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i].reshape(batch_size2, self.m)
            for j in range(batch_size2):
                pred_ij = pred_i[j,:].reshape(-1,)
                idx = split_idx_i[j,:].reshape(-1,)
                out[idx] = out[idx] + pred_ij

        out = out / self.l

        # out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        # out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1,)


# generate multi-level graph, with split and assemble
class RandomMultiMeshSplitter(object):
    def __init__(self, real_space, mesh_size, level, sample_sizes):
        super(RandomMultiMeshSplitter, self).__init__()

        self.d = len(real_space)
        self.ms = sample_sizes
        self.m = sample_sizes[0]
        self.level = level

        assert len(sample_sizes) == level
        assert len(mesh_size) == self.d

        if self.d == 1:
            self.n = mesh_size[0]
            self.grid = np.linspace(real_space[0][0], real_space[0][1], self.n).reshape((self.n, 1))
        else:
            self.n = 1
            grids = []
            for j in range(self.d):
                grids.append(np.linspace(real_space[j][0], real_space[j][1], mesh_size[j]))
                self.n *= mesh_size[j]

            self.grid = np.vstack([xx.ravel() for xx in np.meshgrid(*grids)]).T

        self.splits = self.n // self.m # number of sub-grid
        if self.splits * self.m < self.n:
            self.splits = self.splits + 1
        print('n:',self.n,' m:',self.m, ' number of splits:', self.splits )

        self.perm = None
        self.idx = []
        self.idx_all = None
        self.grid_sample = []
        self.grid_sample_all = None
        self.edge_index = []
        self.edge_index_down = []
        self.edge_index_up = []
        self.edge_attr = []
        self.edge_attr_down = []
        self.edge_attr_up = []
        self.n_edges_inner = []
        self.n_edges_inter = []

    def sample(self, new_sample=True, index0=0):
        self.idx = []
        self.grid_sample = []

        if (new_sample) or (self.perm is None):
            self.perm = torch.randperm(self.n)

        index = index0
        for l in range(self.level):
            index =  index % self.n
            index_end = (index+self.ms[l]) % self.n

            if index < index_end:
                idx = self.perm[index: index_end]
            else:
                idx = torch.cat((self.perm[index: ],self.perm[: index_end]), dim=0)
            self.idx.append(idx)
            self.grid_sample.append(self.grid[idx])
            index = index_end

        if index0 < index_end:
            idx_all = self.perm[index0: index_end]
        else:
            idx_all = torch.cat((self.perm[index0:], self.perm[: index_end]), dim=0)

        self.idx_all = idx_all
        self.grid_sample_all = self.grid[self.idx_all]
        return self.idx, self.idx_all

    def get_grid(self):
        grid_out = []
        for grid in self.grid_sample:
            grid_out.append(torch.tensor(grid, dtype=torch.float))
        return grid_out, torch.tensor(self.grid_sample_all, dtype=torch.float)

    def ball_connectivity(self, radius_inner, radius_inter):
        assert len(radius_inner) == self.level
        assert len(radius_inter) == self.level - 1

        self.edge_index = []
        self.edge_index_down = []
        self.edge_index_up = []
        self.n_edges_inner = []
        self.n_edges_inter = []
        edge_index_out = []
        edge_index_down_out = []
        edge_index_up_out = []

        index = 0
        for l in range(self.level):
            pwd = sklearn.metrics.pairwise_distances(self.grid_sample[l])
            edge_index = np.vstack(np.where(pwd <= radius_inner[l])) + index
            self.edge_index.append(edge_index)
            edge_index_out.append(torch.tensor(edge_index, dtype=torch.long))
            self.n_edges_inner.append(edge_index.shape[1])
            index = index + self.grid_sample[l].shape[0]

        index = 0
        for l in range(self.level-1):
            pwd = sklearn.metrics.pairwise_distances(self.grid_sample[l], self.grid_sample[l+1])
            edge_index = np.vstack(np.where(pwd <= radius_inter[l])) + index
            edge_index[1, :] = edge_index[1, :] + self.grid_sample[l].shape[0]
            self.edge_index_down.append(edge_index)
            edge_index_down_out.append(torch.tensor(edge_index, dtype=torch.long))
            self.edge_index_up.append(edge_index[[1,0],:])
            edge_index_up_out.append(torch.tensor(edge_index[[1,0],:], dtype=torch.long))
            self.n_edges_inter.append(edge_index.shape[1])
            index = index + self.grid_sample[l].shape[0]

        edge_index_out = torch.cat(edge_index_out, dim=1)
        edge_index_down_out = torch.cat(edge_index_down_out, dim=1)
        edge_index_up_out = torch.cat(edge_index_up_out, dim=1)

        return edge_index_out, edge_index_down_out, edge_index_up_out

    def get_edge_index_range(self):
        # in order to use graph network's data structure,
        # the edge index shall be stored as tensor instead of list
        # we concatenate the edge index list and label the range of each level

        edge_index_range = torch.zeros((self.level,2), dtype=torch.long)
        edge_index_down_range = torch.zeros((self.level-1,2), dtype=torch.long)
        edge_index_up_range = torch.zeros((self.level-1,2), dtype=torch.long)

        n_edge_index = 0
        for l in range(self.level):
            edge_index_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + self.edge_index[l].shape[1]
            edge_index_range[l, 1] = n_edge_index

        n_edge_index = 0
        for l in range(self.level-1):
            edge_index_down_range[l, 0] = n_edge_index
            edge_index_up_range[l, 0] = n_edge_index
            n_edge_index = n_edge_index + self.edge_index_down[l].shape[1]
            edge_index_down_range[l, 1] = n_edge_index
            edge_index_up_range[l, 1] = n_edge_index

        return edge_index_range, edge_index_down_range, edge_index_up_range

    def attributes(self, theta=None):
        self.edge_attr = []
        self.edge_attr_down = []
        self.edge_attr_up = []

        if theta is None:
            for l in range(self.level):
                edge_attr = self.grid_sample_all[self.edge_index[l].T].reshape((self.n_edges_inner[l], 2*self.d))
                self.edge_attr.append(torch.tensor(edge_attr))

            for l in range(self.level - 1):
                edge_attr_down = self.grid_sample_all[self.edge_index_down[l].T].reshape((self.n_edges_inter[l], 2*self.d))
                edge_attr_up = self.grid_sample_all[self.edge_index_up[l].T].reshape((self.n_edges_inter[l], 2*self.d))
                self.edge_attr_down.append(torch.tensor(edge_attr_down))
                self.edge_attr_up.append(torch.tensor(edge_attr_up))
        else:
            theta = theta[self.idx_all]

            for l in range(self.level):
                edge_attr = np.zeros((self.n_edges_inner[l], 2 * self.d + 2))
                edge_attr[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index[l].T].reshape(
                    (self.n_edges_inner[l], 2 * self.d))
                edge_attr[:, 2 * self.d] = theta[self.edge_index[l][0]]
                edge_attr[:, 2 * self.d + 1] = theta[self.edge_index[l][1]]
                self.edge_attr.append(torch.tensor(edge_attr, dtype=torch.float))

            for l in range(self.level - 1):
                edge_attr_down = np.zeros((self.n_edges_inter[l], 2 * self.d + 2))
                edge_attr_up = np.zeros((self.n_edges_inter[l], 2 * self.d + 2))

                edge_attr_down[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index_down[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_down[:, 2 * self.d] = theta[self.edge_index_down[l][0]]
                edge_attr_down[:, 2 * self.d + 1] = theta[self.edge_index_down[l][1]]
                self.edge_attr_down.append(torch.tensor(edge_attr_down, dtype=torch.float))

                edge_attr_up[:, 0:2 * self.d] = self.grid_sample_all[self.edge_index_up[l].T].reshape(
                    (self.n_edges_inter[l], 2 * self.d))
                edge_attr_up[:, 2 * self.d] = theta[self.edge_index_up[l][0]]
                edge_attr_up[:, 2 * self.d + 1] = theta[self.edge_index_up[l][1]]
                self.edge_attr_up.append(torch.tensor(edge_attr_up, dtype=torch.float))

        edge_attr_out = torch.cat(self.edge_attr, dim=0)
        edge_attr_down_out = torch.cat(self.edge_attr_down, dim=0)
        edge_attr_up_out = torch.cat(self.edge_attr_up, dim=0)
        return edge_attr_out, edge_attr_down_out, edge_attr_up_out

    def splitter(self, radius_inner, radius_inter, theta_a, theta_all):
        # give a test mesh, generate a list of data
        data = []
        index = 0
        for i in range(self.splits):
            if i==0:
                idx, idx_all = self.sample(new_sample=True, index0=index)
            else:
                idx, idx_all = self.sample(new_sample=False, index0=index)
            index = (index + self.m) % self.n

            grid, grid_all = self.get_grid()
            edge_index, edge_index_down, edge_index_up = self.ball_connectivity(radius_inner, radius_inter)
            edge_index_range, edge_index_down_range, edge_index_up_range = self.get_edge_index_range()
            edge_attr, edge_attr_down, edge_attr_up = self.attributes(theta=theta_a)
            x = torch.cat([grid_all, theta_all[idx_all,:] ], dim=1)
            data.append(Data(x=x,
                  edge_index_mid=edge_index, edge_index_down=edge_index_down, edge_index_up=edge_index_up,
                  edge_index_range=edge_index_range, edge_index_down_range=edge_index_down_range, edge_index_up_range=edge_index_up_range,
                  edge_attr_mid=edge_attr, edge_attr_down=edge_attr_down, edge_attr_up=edge_attr_up,
                  sample_idx=idx[0]))
        return data

    def assembler(self, out_list, sample_idx_list, is_cuda=False):
        assert len(out_list) == self.splits
        if is_cuda:
            pred = torch.zeros(self.n, ).cuda()
        else:
            pred = torch.zeros(self.n, )
        for i in range(self.splits):
            pred[sample_idx_list[i]] = out_list[i].reshape(-1)
        return pred

# generate graph, with split and assemble with downsample
class DownsampleGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15,  edge_features=1):
        super(DownsampleGridSplitter, self).__init__()

        # instead of randomly sample sub-grids, here we downsample sub-grids

        self.grid = grid.reshape(resolution, resolution,2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution%2==1:
            self.s = int(((resolution - 1)/r) + 1)
        else:
            self.s = int(resolution/r)
        self.r = r
        self.n = resolution**2
        self.m = m
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(self.resolution, self.resolution)

    def ball_connectivity(self, grid):
        pwd = sklearn.metrics.pairwise_distances(grid)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges

    def get_data(self, theta):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x::self.r, y::self.r,:].reshape(-1,2)
                theta_sub = theta[x::self.r, y::self.r,:].reshape(-1,theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n,-1)[idx]
                theta_sample = theta.reshape(self.n,-1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample],dim=0)
                theta_split = torch.cat([theta_sub, theta_sample],dim=0)
                X = torch.cat([grid_split,theta_split],dim=1)

                edge_index, n_edges = self.ball_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 4+self.edge_features*2))
                a = theta_split[:, :self.edge_features]
                edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
                edge_attr[:, 4:4 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 4 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x,y],dtype=torch.long).reshape(1,2)

                data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0,self.r,(1,))
        y = torch.randint(0,self.r,(1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x::self.r, y::self.r].reshape(-1,)
        index_sub = self.index[x::self.r, y::self.r].reshape(-1,)
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.n, )[idx]


            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(-1,)
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1,)
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(-1,)
            index_split = index_sub.reshape(-1,)
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges = self.ball_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 4+self.edge_features*2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, :4] = grid_split[edge_index.T].reshape(n_edges, -1)
        edge_attr[:, 4:4+self.edge_features] = a[edge_index[0]]
        edge_attr[:, 4+self.edge_features: 4+self.edge_features*2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx, sample_idx=index_split)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data


    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r**2 // batch_size2

        out = torch.zeros((self.resolution,self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j,:]
                x, y = split_idx_i[j]
                if self.resolution%2==1:
                    if x==0:
                        nx = self.s
                    else:
                        nx = self.s-1
                    if y==0:
                        ny = self.s
                    else:
                        ny = self.s-1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx,ny)

        out = gaussian_filter(out, sigma=sigma, mode='constant', cval=0)
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1,)

# generate graph on Torus, with split and assemble
class TorusGridSplitter(object):
    def __init__(self, grid, resolution, r, m=100, radius=0.15, T=None, edge_features=1, ):
        super(TorusGridSplitter, self).__init__()

        self.grid = grid.reshape(resolution, resolution,2)
        # self.theta = theta.reshape(resolution, resolution,-1)
        # self.y = y.reshape(resolution, resolution,1)
        self.resolution = resolution
        if resolution%2==1:
            self.s = int(((resolution - 1)/r) + 1)
        else:
            self.s = int(resolution/r)
        self.r = r
        self.n = resolution**2
        self.m = m
        self.T = T
        self.radius = radius
        self.edge_features = edge_features

        self.index = torch.tensor(range(self.n), dtype=torch.long).reshape(self.resolution, self.resolution)

    def pairwise_difference(self,grid1, grid2):
        n = grid1.shape[0]
        x1 = grid1[:,0]
        y1 = grid1[:,1]
        x2 = grid2[:,0]
        y2 = grid2[:,1]

        X1 = np.tile(x1.reshape(n, 1), [1, n])
        X2 = np.tile(x2.reshape(1, n), [n, 1])
        X_diff = X1 - X2
        Y1 = np.tile(y1.reshape(n, 1), [1, n])
        Y2 = np.tile(y2.reshape(1, n), [n, 1])
        Y_diff = Y1 - Y2

        return X_diff, Y_diff

    def torus_connectivity(self, grid):
        pwd0 = sklearn.metrics.pairwise_distances(grid, grid)
        X_diff0, Y_diff0 = self.pairwise_difference(grid, grid)

        grid1 = grid
        grid1[:,0] = grid[:,0]+1
        pwd1 = sklearn.metrics.pairwise_distances(grid, grid1)
        X_diff1, Y_diff1 = self.pairwise_difference(grid, grid1)

        grid2 = grid
        grid2[:, 1] = grid[:, 1] + 1
        pwd2 = sklearn.metrics.pairwise_distances(grid, grid2)
        X_diff2, Y_diff2 = self.pairwise_difference(grid, grid2)

        grid3 = grid
        grid3[:, :] = grid[:, :] + 1
        pwd3 = sklearn.metrics.pairwise_distances(grid, grid3)
        X_diff3, Y_diff3 = self.pairwise_difference(grid, grid3)

        grid4 = grid
        grid4[:, 0] = grid[:, 0] + 1
        grid4[:, 1] = grid[:, 1] - 1
        pwd4 = sklearn.metrics.pairwise_distances(grid, grid4)
        X_diff4, Y_diff4 = self.pairwise_difference(grid, grid4)

        PWD = np.stack([pwd0,pwd1,pwd2,pwd3,pwd4], axis=2)
        X_DIFF = np.stack([X_diff0,X_diff1,X_diff2,X_diff3,X_diff4], axis=2)
        Y_DIFF = np.stack([Y_diff0, Y_diff1, Y_diff2, Y_diff3, Y_diff4], axis=2)
        pwd = np.min(PWD, axis=2)
        pwd_index = np.argmin(PWD, axis=2)
        edge_index = np.vstack(np.where(pwd <= self.radius))
        pwd_index =  pwd_index[np.where(pwd <= self.radius)]
        PWD_index = (np.where(pwd <= self.radius)[0],  np.where(pwd <= self.radius)[1], pwd_index)
        distance = PWD[PWD_index]
        X_difference = X_DIFF[PWD_index]
        Y_difference = Y_DIFF[PWD_index]
        n_edges = edge_index.shape[1]
        return torch.tensor(edge_index, dtype=torch.long), n_edges, distance, X_difference, Y_difference


    def get_data(self, theta, params=None):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        data = []
        for x in range(self.r):
            for y in range(self.r):
                grid_sub = self.grid[x::self.r, y::self.r,:].reshape(-1,2)
                theta_sub = theta[x::self.r, y::self.r,:].reshape(-1,theta_d)

                perm = torch.randperm(self.n)
                m = self.m - grid_sub.shape[0]
                idx = perm[:m]
                grid_sample = self.grid.reshape(self.n,-1)[idx]
                theta_sample = theta.reshape(self.n,-1)[idx]

                grid_split = torch.cat([grid_sub, grid_sample],dim=0)
                theta_split = torch.cat([theta_sub, theta_sample],dim=0)
                X = torch.cat([grid_split,theta_split],dim=1)

                edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

                edge_attr = np.zeros((n_edges, 3+self.edge_features*2))
                a = theta_split[:, :self.edge_features]
                edge_attr[:, 0] = X_difference.reshape(n_edges, )
                edge_attr[:, 1] = Y_difference.reshape(n_edges, )
                edge_attr[:, 2] = distance.reshape(n_edges, )
                edge_attr[:, 3:3 + self.edge_features] = a[edge_index[0]]
                edge_attr[:, 3 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
                edge_attr = torch.tensor(edge_attr, dtype=torch.float)
                split_idx = torch.tensor([x,y],dtype=torch.long).reshape(1,2)

                if params==None:
                    data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx))
                else:
                    data.append(Data(x=X, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx, params=params))
        print('test', len(data), X.shape, edge_index.shape, edge_attr.shape)
        return data

    def sample(self, theta, Y):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.resolution, self.resolution)

        x = torch.randint(0,self.r,(1,))
        y = torch.randint(0,self.r,(1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[x::self.r, y::self.r].reshape(-1,)
        index_sub = self.index[x::self.r, y::self.r].reshape(-1,)
        n_sub = Y_sub.shape[0]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.n, )[idx]


            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=0).reshape(-1,)
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1,)
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(-1,)
            index_split = index_sub.reshape(-1,)
            X = torch.cat([grid_split, theta_split], dim=1)


        edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 3+self.edge_features*2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, 0] = X_difference.reshape(n_edges, )
        edge_attr[:, 1] = Y_difference.reshape(n_edges, )
        edge_attr[:, 2] = distance.reshape(n_edges, )
        edge_attr[:, 3:3+self.edge_features] = a[edge_index[0]]
        edge_attr[:, 3+self.edge_features: 4+self.edge_features*2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx, sample_idx=index_split)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data

    def sampleT(self, theta, Y, params=None):
        theta_d = theta.shape[1]
        theta = theta.reshape(self.resolution, self.resolution, theta_d)
        Y = Y.reshape(self.T, self.resolution, self.resolution)

        x = torch.randint(0, self.r, (1,))
        y = torch.randint(0, self.r, (1,))

        grid_sub = self.grid[x::self.r, y::self.r, :].reshape(-1, 2)
        theta_sub = theta[x::self.r, y::self.r, :].reshape(-1, theta_d)
        Y_sub = Y[:,x::self.r, y::self.r].reshape(self.T,-1)
        index_sub = self.index[x::self.r, y::self.r].reshape(-1, )
        n_sub = Y_sub.shape[1]

        if self.m >= n_sub:
            m = self.m - n_sub
            perm = torch.randperm(self.n)
            idx = perm[:m]
            grid_sample = self.grid.reshape(self.n, -1)[idx]
            theta_sample = theta.reshape(self.n, -1)[idx]
            Y_sample = Y.reshape(self.T, self.n)[:,idx]

            grid_split = torch.cat([grid_sub, grid_sample], dim=0)
            theta_split = torch.cat([theta_sub, theta_sample], dim=0)
            Y_split = torch.cat([Y_sub, Y_sample], dim=1).reshape(self.T,-1)
            index_split = torch.cat([index_sub, idx], dim=0).reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        else:
            grid_split = grid_sub
            theta_split = theta_sub
            Y_split = Y_sub.reshape(self.T, -1)
            index_split = index_sub.reshape(-1, )
            X = torch.cat([grid_split, theta_split], dim=1)

        edge_index, n_edges, distance, X_difference, Y_difference = self.torus_connectivity(grid_split)

        edge_attr = np.zeros((n_edges, 3 + self.edge_features * 2))
        a = theta_split[:, :self.edge_features]
        edge_attr[:, 0] = X_difference.reshape(n_edges, )
        edge_attr[:, 1] = Y_difference.reshape(n_edges, )
        edge_attr[:, 2] = distance.reshape(n_edges, )
        edge_attr[:, 3:3 + self.edge_features] = a[edge_index[0]]
        edge_attr[:, 3 + self.edge_features: 4 + self.edge_features * 2] = a[edge_index[1]]
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        split_idx = torch.tensor([x, y], dtype=torch.long).reshape(1, 2)
        if params==None:
            data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx,
                        sample_idx=index_split)
        else:
            data = Data(x=X, y=Y_split, edge_index=edge_index, edge_attr=edge_attr, split_idx=split_idx,
                        sample_idx=index_split, params=params)
        print('train', X.shape, Y_split.shape, edge_index.shape, edge_attr.shape, index_split.shape)

        return data

    def assemble(self, pred, split_idx, batch_size2, sigma=1):
        assert len(pred) == len(split_idx)
        assert len(pred) == self.r**2 // batch_size2

        out = torch.zeros((self.resolution,self.resolution))
        for i in range(len(pred)):
            pred_i = pred[i].reshape(batch_size2, self.m)
            split_idx_i = split_idx[i]
            for j in range(batch_size2):
                pred_ij = pred_i[j,:]
                x, y = split_idx_i[j]
                if self.resolution%2==1:
                    if x==0:
                        nx = self.s
                    else:
                        nx = self.s-1
                    if y==0:
                        ny = self.s
                    else:
                        ny = self.s-1
                else:
                    nx = self.s
                    ny = self.s
                # pred_ij = pred_i[idx : idx + nx * ny]
                out[x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx,ny)

        out = gaussian_filter(out, sigma=sigma, mode='wrap')
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(-1,)

    def assembleT(self, pred, split_idx, batch_size2, sigma=1):
        # pred is a list (batches) of list (time seq)
        assert len(pred) == len(split_idx)
        assert len(pred[0]) == self.T
        assert len(pred) == self.r**2 // batch_size2

        out = torch.zeros((self.T, self.resolution,self.resolution))
        for t in range(self.T):
            for i in range(len(pred)):
                pred_i = pred[i][t].reshape(batch_size2, self.m)
                split_idx_i = split_idx[i]
                for j in range(batch_size2):
                    pred_ij = pred_i[j,:]
                    x, y = split_idx_i[j]
                    if self.resolution%2==1:
                        if x==0:
                            nx = self.s
                        else:
                            nx = self.s-1
                        if y==0:
                            ny = self.s
                        else:
                            ny = self.s-1
                    else:
                        nx = self.s
                        ny = self.s
                    # pred_ij = pred_i[idx : idx + nx * ny]
                    out[t, x::self.r, y::self.r] = pred_ij[:nx * ny].reshape(nx,ny)

        out = gaussian_filter(out, sigma=sigma, mode='wrap')
        out = torch.tensor(out, dtype=torch.float)
        return out.reshape(self.T,self.n)


def downsample(data, grid_size, l):
    data = data.reshape(-1, grid_size, grid_size)
    data = data[:, ::l, ::l]
    data = data.reshape(-1, (grid_size // l) ** 2)
    return data


def simple_grid(n_x, n_y):
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
            if (x != n_x - 1):
                edge_index.append((i, i + 1))
                edge_attr.append((1, 0, 0))
                edge_index.append((i + 1, i))
                edge_attr.append((-1, 0, 0))

            if (y != n_y - 1):
                edge_index.append((i, i + n_x))
                edge_attr.append((0, 1, 0))
                edge_index.append((i + n_x, i))
                edge_attr.append((0, -1, 0))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr


def grid_edge(n_x, n_y, a=None):
    if a != None:
        a = a.reshape(n_x, n_y)

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
            if (x != n_x - 1):
                d = 1 / n_x
                edge_index.append((i, i + 1))
                edge_index.append((i + 1, i ))
                if a != None:
                    a1 = a[x, y]
                    a2 = a[x + 1, y]
                    edge_attr.append((x / n_x, y / n_y, a1, a2))
                    edge_attr.append((y/n_y, x/n_x, a2, a1))

            if (y != n_y - 1):
                d = 1 / n_y
                edge_index.append((i, i + n_x))
                edge_index.append((i + n_x, i))
                if a != None:
                    a1 = a[x, y]
                    a2 = a[x, y+1]
                    edge_attr.append((x/n_x, y/n_y, a1, a2))
                    edge_attr.append((y/n_y, x/n_x, a2, a1))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr

def grid_edge1d(n_x, a=None):
    if a != None:
        a = a.reshape(n_x)

    xs = np.linspace(0.0, 1.0, n_x)
    # xs = np.array(range(n_x))
    # ys = np.array(range(n_y))

    edge_index = []
    edge_attr = []
    for x in range(n_x):
        i = x

        i1 = (x+1)%n_x
        edge_index.append((i, i1))
        edge_index.append((i1, i ))

        i2 = (x + 2) % n_x
        edge_index.append((i, i2))
        edge_index.append((i2, i ))

        if a != None:
            a1 = a[x]
            a2 = a[x + 1]
            edge_attr.append((x / n_x, a1, a2))
            edge_attr.append((x / n_x, a2, a1))

    X = torch.tensor(xs, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr

def grid_edge_aug(n_x, n_y, a):
    a = a.reshape(n_x, n_y)
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
            if (x != n_x - 1):
                d = 1 / n_x
                a1 = a[x, y]
                a2 = a[x + 1, y]
                edge_index.append((i, i + 1))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                 np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i + 1, i))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

            if (y != n_y - 1):
                d = 1 / n_y
                a1 = a[x, y]
                a2 = a[x, y+1]
                edge_index.append((i, i + n_x))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i + n_x, i))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr

def grid_edge_aug_full(n_x, n_y, r, a):
    n = n_x * n_y

    xs = np.linspace(0.0, 1.0, n_x)
    ys = np.linspace(0.0, 1.0, n_y)

    grid = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T

    edge_index = []
    edge_attr = []

    for i1 in range(n):
        x1 = grid[i1]
        for i2 in range(n):
            x2 = grid[i2]

            d = np.linalg.norm(x1-x2)

            if(d<=r):
                a1 = a[i1]
                a2 = a[i2]
                edge_index.append((i1, i2))
                edge_attr.append((d, a1, a2, 1 / np.sqrt(np.abs(a1 * a2)),
                                 np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))
                edge_index.append((i2, i1))
                edge_attr.append((d, a2, a1, 1 / np.sqrt(np.abs(a1 * a2)),
                                  np.exp(-(d) ** 2), np.exp(-(d / 0.1) ** 2), np.exp(-(d / 0.01) ** 2)))

    X = torch.tensor(grid, dtype=torch.float)
    # Exact = torch.tensor(Exact, dtype=torch.float).view(-1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).transpose(0, 1)
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    return X, edge_index, edge_attr

def multi_grid(depth, n_x, n_y, grid, params):

    edge_index_global = []
    edge_attr_global = []
    X_global = []
    num_nodes = 0

    # build connected graph
    for l in range(depth):
        h_x_l = n_x // (2 ** l)
        h_y_l = n_y // (2 ** l)
        n_l = h_x_l * h_y_l

        a = downsample(params, n_x, (2 ** l))
        if grid == 'grid':
            X, edge_index_inner, edge_attr_inner = grid(h_y_l, h_x_l)
        elif grid == 'grid_edge':
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)
        elif grid == 'grid_edge_aug':
            X, edge_index_inner, edge_attr_inner = grid_edge(h_y_l, h_x_l, a)

        # update index
        edge_index_inner = edge_index_inner + num_nodes
        edge_index_global.append(edge_index_inner)
        edge_attr_global.append(edge_attr_inner)

        # construct X
        # if (is_high):
        #     X = torch.cat([torch.zeros(n_l, l * 2), X, torch.zeros(n_l, (depth - 1 - l) * 2)], dim=1)
        # else:
        #     X_l = torch.tensor(l, dtype=torch.float).repeat(n_l, 1)
        #     X = torch.cat([X, X_l], dim=1)
        X_global.append(X)

        # construct edges
        index1 = torch.tensor(range(n_l), dtype=torch.long)
        index1 = index1 + num_nodes
        num_nodes += n_l

        # #construct inter-graph edge
        if l != depth-1:
            index2 = np.array(range(n_l//4)).reshape(h_x_l//2, h_y_l//2)  # torch.repeat is different from numpy
            index2 = index2.repeat(2, axis = 0).repeat(2, axis = 1)
            index2 = torch.tensor(index2).reshape(-1)
            index2 = index2 + num_nodes
            index2 = torch.tensor(index2, dtype=torch.long)

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
    # print('create multi_grid with size:', X.shape,  edge_index.shape, edge_attr.shape, mask_index.shape)

    return (X, edge_index, edge_attr, mask_index, num_nodes)


def multi_pole_grid1d(theta, theta_d, s,  N, is_periodic=False):
    grid_list = []
    theta_list = []
    edge_index_list = []
    edge_index_list_cuda = []
    level = int(np.log2(s) - 1)
    print(level)
    for l in range(1, level+1):
        r_l = 2 ** (l - 1)
        s_l = s // r_l
        n_l = s_l
        print('level',s_l,r_l,n_l)
        xs = np.linspace(0.0, 1.0, s_l)
        grid_l = xs
        grid_l = torch.tensor(grid_l, dtype=torch.float)
        print(grid_l.shape)
        grid_list.append(grid_l)

        theta_l = theta[:,:,:theta_d].reshape(N, s, theta_d)
        theta_l = theta_l[:, ::r_l,  :]
        theta_l = theta_l.reshape(N, n_l, theta_d)
        theta_l = torch.tensor(theta_l, dtype=torch.float)
        print(theta_l.shape)
        theta_list.append(theta_l)

        # for the finest level, we construct the nearest neighbors (NN)
        if l==1:
            edge_index_nn = []
            for x_i in range(s_l):
                for x in (-1,1):
                    x_j = x_i + x

                    if is_periodic:
                        x_j = x_j % s_l

                    # if (xj, yj) is a valid node
                    if (x_j in range(s_l)):
                        edge_index_nn.append([x_i,x_j])
            edge_index_nn = torch.tensor(edge_index_nn, dtype=torch.long)
            edge_index_nn = edge_index_nn.transpose(0,1)
            edge_index_list.append(edge_index_nn)
            edge_index_list_cuda.append(edge_index_nn.cuda())
            print('edge', edge_index_nn.shape)

        # we then compute the interactive neighbors -- their parents are NN but they are not NearestNeighbor
        edge_index_inter = []
        for x_i in range(s_l):
            for x in range(-3,4):
                x_j = x_i + x
                # if (xj, yj) is a valid node
                if is_periodic:
                    x_j = x_j % s_l

                if (x_j in range(s_l)):
                    # if (xi, yi), (xj, yj) not NearestNeighbor
                    if abs(x)>=2:
                        # if their parents are NN
                        if abs(x_i//2 - x_j//2)%(s_l//2) <=1:
                            edge_index_inter.append([x_i,x_j])

        edge_index_inter = torch.tensor(edge_index_inter, dtype=torch.long)
        edge_index_inter = edge_index_inter.transpose(0,1)
        edge_index_list.append(edge_index_inter)
        edge_index_list_cuda.append(edge_index_inter.cuda())
        print('edge_inter', edge_index_inter.shape)

    print(len(grid_list),len(edge_index_list),len(theta_list))
    return grid_list, theta_list, edge_index_list, edge_index_list_cuda

def get_edge_attr(grid, theta, edge_index):
    n_edges = edge_index.shape[1]
    edge_attr = np.zeros((n_edges, 4))
    edge_attr[:, 0:2] = grid[edge_index.transpose(0,1)].reshape((n_edges, -1))
    edge_attr[:, 2] = theta[edge_index[0]]
    edge_attr[:, 3] = theta[edge_index[1]]
    return torch.tensor(edge_attr, dtype=torch.float)
