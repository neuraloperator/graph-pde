import torch
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from utilities import *
from nn_conv import NNConv, NNConv_old

from timeit import default_timer
import scipy.io

class KernelNN(torch.nn.Module):
    def __init__(self, width,  depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, width//4, width], torch.nn.ReLU)
        self.conv1 = NNConv(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x

class KernelNN2(torch.nn.Module):
    def __init__(self, width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN2, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, width//4, width], torch.nn.ReLU)
        self.conv1 = NNConv(width, width, kernel, aggr='mean')

        kernel2 = DenseNet([ker_in, width * in_width], torch.nn.ReLU)
        self.conv2 = NNConv_old(in_width, width, kernel2, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x

class KernelNN3(torch.nn.Module):
    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN3, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)

        kernel = DenseNet([ker_in, width_kernel // 4, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width_node, width_node, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr)) + self.fc1(data.x)

        x = self.fc2(x)
        return x

TRAIN_PATH = 'data/piececonst_r121_N1024.mat'
TEST_PATH = 'data/piececonst_r121_N10000.mat'
INIT_PATH = 'data/poisson_r121_f1.mat'


ntrain = 100
ntest = 10
epochs = 50
batch_size = 10
batch_size2 = 10
width_node = 32
width_kernel = 1000
depth = 6
edge_features = 6

r = 1

s = int(((121 - 1)/r) + 1)
n = s**2
m = 200
k = 1
radius = 0.15
radius2 = 0.15
print('resolution', s)

learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.8


path_train_err = 'results/LargeKernel_s'+str(s)+'_n'+ str(ntrain)+'_m'+ str(m)+'_r'+ str(radius) + 'train.txt'
path_test_err = 'results/LargeKernel_s'+str(s)+'_n'+ str(ntrain)+'_m'+ str(m)+'_r'+ str(radius) + 'test.txt'
path_image = 'results/LargeKernel_s'+str(s)+'_n'+ str(ntrain)+'_m'+ str(m)+'_r'+ str(radius) + '_'




ttrain = np.zeros((epochs, ))
ttest = np.zeros((epochs,))

t1 = default_timer()


reader = MatReader(TRAIN_PATH)
train_x = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_y = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

reader.load_file(TRAIN_PATH)
test_x = reader.read_field('coeff')[-ntest:,::r,::r].reshape(ntest,-1)
test_y = reader.read_field('sol')[-ntest:,::r,::r].reshape(ntest,-1)

reader.load_file(INIT_PATH)
init_point = reader.read_field('sol')[::r,::r].reshape(-1)



x_normalizer = UnitGaussianNormalizer(train_x)
train_x = x_normalizer.encode(train_x)
test_x = x_normalizer.encode(test_x)

y_normalizer = UnitGaussianNormalizer(train_y)
train_y = y_normalizer.encode(train_y)
# test_y = y_normalizer.encode(test_y)


# meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[s,s])
# edge_index = meshgenerator.ball_connectivity(radius2)
# grid = meshgenerator.get_grid()
# data_train = []
# for j in range(ntrain):
#     edge_attr = meshgenerator.attributes(theta=train_x[j,:])
#     #data_test.append(Data(x=init_point.clone().view(-1,1), y=test_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
#     data_train.append(Data(x=torch.cat([grid, train_x[j,:].reshape(-1,1)], dim=1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))

meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[s,s], sample_size=m)
data_train = []
for j in range(ntrain):
    for i in range(k):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius)
        edge_attr = meshgenerator.attributes(theta=train_x[j,:])
        #data_train.append(Data(x=init_point.clone().view(-1,1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
        data_train.append(Data(x=torch.cat([grid, train_x[j,idx].reshape(-1,1)], dim=1), y=train_y[j,idx], edge_index=edge_index, edge_attr=edge_attr))
        print(j,i, 'grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)


# meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[s,s])
# edge_index = meshgenerator.ball_connectivity(radius2)
# grid = meshgenerator.get_grid()
# data_test = []
# for j in range(ntest):
#     edge_attr = meshgenerator.attributes(theta=test_x[j,:])
#     #data_test.append(Data(x=init_point.clone().view(-1,1), y=test_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
#     data_test.append(Data(x=torch.cat([grid, test_x[j,:].reshape(-1,1)], dim=1), y=test_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
# print('grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)

meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[s,s], sample_size=m)
data_test = []
for j in range(ntest):
    for i in range(k):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius2)
        edge_attr = meshgenerator.attributes(theta=test_x[j,:])
        data_test.append(Data(x=torch.cat([grid, test_x[j,idx].reshape(-1,1)], dim=1), y=test_y[j,idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx = idx))
        print(j,i, 'grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
#
#
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

# model = KernelNN(width,depth,edge_features,in_width=3).cuda()
model = KernelNN3(width_node,width_kernel,depth,edge_features,in_width=3).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()

model.train()
for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0.0
    train_l2 = 0.0
    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
        mse.backward()

        l2 = myloss(out.view(batch_size,-1), batch.y.view(batch_size, -1))

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            out = y_normalizer.decode(out.view(batch_size2,-1), sample_idx=batch.sample_idx.view(batch_size2,-1))
            test_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()
            # test_l2 += myloss(out.view(batch_size2,-1), y_normalizer.encode(batch.y.view(batch_size2, -1))).item()

    ttrain[ep] = train_l2/(ntrain * k)
    ttest[ep] = test_l2/ntest

    print(ep, t2-t1, train_mse/len(train_loader), train_l2/(ntrain * k), test_l2/ntest)

np.savetxt(path_train_err, ttrain)
np.savetxt(path_test_err, ttest)

plt.figure()
plt.plot(ttrain, label='train loss')
plt.plot(ttest, label='test loss')
plt.legend(loc='upper right')
plt.show()
