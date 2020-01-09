import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv, NNConv
from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt
import random
import scipy.io
import h5py
import pickle

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

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, 32)
        self.conv1 = GCNConv(32, 32)
        self.conv2 = GCNConv(32, 32)
        self.conv3 = GCNConv(32, 32)
        self.fc2 = nn.Linear(32, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.fc2(x)

        return x



class UnitGaussianNormalizer(object):
    def __init__(self, x, zeroone_first=True):
        super(UnitGaussianNormalizer, self).__init__()
        self.zeroone_first = zeroone_first
        if self.zeroone_first:
            self.min = torch.min(x, 0)[0].view(-1, )
            self.max = torch.max(x, 0)[0].view(-1, )

            s = x.size()
            x = ((x.view(s[0], -1) - self.min) / (self.max - self.min)).view(s)

        self.mean = torch.mean(x, 0).view(-1, )
        self.std = torch.std(x, 0).view(-1, )

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        if self.zeroone_first:
            x = (x - self.min) / (self.max - self.min)
        x = (x - self.mean) / self.std
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x * self.std) + self.mean
        if self.zeroone_first:
            x = (x * (self.max - self.min)) + self.min
        x = x.view(s)
        return x


#################################################
#
# generate PDE data
#
#################################################


# # 2*2 mesh on [0,1] * [0,1] domain
#
# # location
# x = torch.tensor([[0,0],[0,1],[1,0],[1,1]], dtype=torch.float)
# # value
# y = torch.tensor([0, 0, 0, 0], dtype=torch.float)
#
# # edge (4 edges)
# edge_index = torch.tensor([[0, 1, 1, 2, 2, 3, 3, 0],
#                            [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long)
#
# dataset = Data(x=x, y=y, edge_index=edge_index)
#
# print(dataset)



# h*h mesh on [0,1] * [0,1] domain
#raw_data = np.loadtxt("/Users/lizongyi/Downloads/GNN-PDE/fenics/mysol00.txt")
#raw_data = np.loadtxt("/Users/lizongyi/Downloads/GNN-PDE/test.txt", delimiter=',').reshape(-1,3)


#np.random.shuffle(raw_data)
# print(raw_data.shape)
# n = raw_data.shape[0]
# h = int(np.sqrt(n))-1
# print('h', h)
# x = torch.tensor(raw_data[:,:2], dtype=torch.float)
# y = torch.tensor(raw_data[:,2], dtype=torch.float)

# counter = 0
# edges = list()
# # generate edge
# for i in range(n):
#     for j in range(i+1,n):
#         if (np.linalg.norm(x[i]-x[j]) <= 1/h):
#             counter = counter + 1
#             print(counter,x[i], x[j])
#             edges.append((i, j))
#             edges.append((j, i))
# edges = np.array(edges).transpose()
# edge_index = torch.tensor(edges, dtype=torch.long)

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

#======================================================================#
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
#======================================================#
TRAIN_PATH = r'C:\Users\Harwa\Dropbox\Postdoc2019\Caltech\Machine_Learning\Project_5_fixed_grain_4_Energy\Trainingdata2DfixedgrainNgrain4Strain8096.mat'
Ntotal     = 8096
num_train = 4000
num_data_per_batch = 64
num_test = Ntotal-num_train

X_TRAIN_FIELD = 'Training_Input_raw'
Y_TRAIN_FIELD = 'Training_Output_raw'
grid_size = 64  # Only for plotting
#data_loader = MatReader(TRAIN_PATH,True,True,True)
data_loader = MatReader(TRAIN_PATH)
data_input  = data_loader.read_field(X_TRAIN_FIELD).contiguous().view(Ntotal, -1)
data_output = data_loader.read_field(Y_TRAIN_FIELD).contiguous().view(Ntotal, -1)

x_train  = data_input[:num_train,:]
x_test   = data_input[num_train:,:]
x_normalizer = UnitGaussianNormalizer(x_train)
x_train_enc = x_normalizer.encode(x_train)
x_test_enc = x_normalizer.encode(x_test)

data_input = torch.cat([x_train_enc,x_test_enc])

y_train = data_input[:num_train, :]
y_test = data_input[num_train:, :]
y_normalizer = UnitGaussianNormalizer(y_train)
y_train_enc = y_normalizer.encode(y_train)
y_test_enc = y_normalizer.encode(y_test)

data_output = torch.cat([y_train_enc,y_test_enc])

X, edge_index, edge_attr = grid(grid_size, grid_size)
dataset = []
for b in range(Ntotal):
    x = torch.tensor(X, dtype=torch.float)
    theta = data_input[b,:]
    x = torch.cat([x,theta.reshape(-1,1)], dim=1)
    y = data_output[b,:]

    edge_index = torch.tensor(edge_index,dtype=torch.long)
    dataset.append(Data(x=x, y=y, edge_index=edge_index))
#==============================================================================#
# number of train data

train_loader = DataLoader(dataset[:num_train], batch_size=num_data_per_batch, shuffle=True)
test_loader  = DataLoader(dataset[num_train:], batch_size=num_data_per_batch, shuffle=False)

#################################################
#
# Graph neural network structure
#
#################################################

class Net_skip(torch.nn.Module):
    def __init__(self):
        super(Net_skip, self).__init__()
        self.conv1 = GCNConv(features, 32-features)
        self.conv2 = GCNConv(32, 32-features)
        self.conv3 = GCNConv(32, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = torch.cat((x,data.x),dim=1)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = torch.cat((x, data.x),dim=1)
        x = self.conv3(x, edge_index)
        return x

#################################################
#
# train
#
#################################################


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Net().to(device)
#model = Net_skip().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


# data = dataset.to(device)
# model.train()
# for epoch in range(100):
#     optimizer.zero_grad()
#     out = model(data)
#     loss = F.mse_loss(out[train_mask], data.y.view(-1,1)[train_mask])
#     print(epoch, loss)
#     loss.backward()
#     optimizer.step()
#
# model.eval()
# pred = model(data)
# error = ((pred[test_mask] - data.y[test_mask])**2).mean()
# print('test L2 error: {:.4f}'.format(error))

test_loss = []
train_loss = []
model.train()
for epoch in range(10):
    train_error = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        #y = torch.cat([data.y for data in batch])
        loss = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
        train_error = train_error + loss
        loss.backward()
        optimizer.step()
    #train_loss.append(train_error / len(train_loader))
    test_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            test_error += F.mse_loss(pred.view(-1, 1), batch.y.view(-1, 1))

    test_loss.append(test_error )
    print(epoch, 'train loss: {:.4f}'.format(train_error/len(train_loader)),
                 'test L2 error: {:.4f}'.format(test_error/len(test_loader)))

model.eval()
test_error = 0
for batch in test_loader:
    batch = batch.to(device)
    pred = model(batch)
    test_error += F.mse_loss(pred, batch.y.view(-1,1))
print('test L2 error: {:.4f}'.format(test_error/len(test_loader)))


#################################################
#
# save
#
#################################################

#torch.save(model, "/Users/lizongyi/Downloads/GNN-PDE/fenics/model")
#torch.save(model.state_dict(), "/Users/lizongyi/Downloads/GNN-PDE/fenics/model")

#################################################
#
# plot
#
#################################################


'''
plt.plot(train_loss, label='train loss')
plt.plot(test_loss, label='test loss')
plt.legend(loc='upper right')
plt.show()
'''
r = np.random.randint(num_data_per_batch)

truth = test_loader.dataset[r].y.detach().cpu().numpy().reshape((grid_size, grid_size))
model.cpu()
approx = model(test_loader.dataset[r]).detach().numpy().reshape((grid_size, grid_size))

plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(truth)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Ground Truth')

plt.subplot(1, 3, 2)
plt.imshow(approx)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Approximation')

plt.subplot(1, 3, 3)
plt.imshow((approx - truth) ** 2)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Error')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
