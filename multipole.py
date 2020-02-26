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
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        X_list,_, edge_index_list, edge_attr_list = data
        level = len(X_list)
        x = X_list[0]
        x = self.fc1(x)

        for k in range(self.depth):
            # upward
            for l in range(level):
                # nn
                # average pooling
                x = F.relu(self.conv1(x, edge_index_list[l + 1], edge_attr_list[l + 1]))
            yn = 0

            #downward
            for l in reversed(range(level)):
                # nn
                # repeat
                x = F.relu(self.conv1(x, edge_index_list[l+1], edge_attr_list[l+1]))

            x = x

        x = self.fc2(x)
        return x


TRAIN_PATH = 'data/piececonst_r65_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r65_N1024_smooth2.mat'

r = 4
s = 16
K = s**2

ntrain = 10
ntest = 10

batch_size = 1
batch_size2 = 1
width = 32
ker_width = 32
depth = 2
edge_features = 6
node_features = 6

epochs = 10
learning_rate = 0.0001
scheduler_step = 50
scheduler_gamma = 0.8

# read data and normalize


reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('coeff')[:ntrain,:64:r,:64:r].reshape(ntrain,-1)
train_a_smooth = reader.read_field('Kcoeff')[:ntrain,:64:r,:64:r].reshape(ntrain,-1)
train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,:64:r,:64:r].reshape(ntrain,-1)
train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,:64:r,:64:r].reshape(ntrain,-1)
train_u = reader.read_field('sol')[:ntrain,:64:r,:64:r].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('coeff')[:ntest,:64:4,:64:4].reshape(ntest,-1)
test_a_smooth = reader.read_field('Kcoeff')[:ntest,:64:4,:64:4].reshape(ntest,-1)
test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,:64:4,:64:4].reshape(ntest,-1)
test_a_grady = reader.read_field('Kcoeff_y')[:ntest,:64:4,:64:4].reshape(ntest,-1)
test_u = reader.read_field('sol')[:ntest,:64:4,:64:4].reshape(ntest,-1)

a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
as_normalizer = GaussianNormalizer(train_a_smooth)
train_a_smooth = as_normalizer.encode(train_a_smooth)
test_a_smooth = as_normalizer.encode(test_a_smooth)
agx_normalizer = GaussianNormalizer(train_a_gradx)
train_a_gradx = agx_normalizer.encode(train_a_gradx)
test_a_gradx = agx_normalizer.encode(test_a_gradx)
agy_normalizer = GaussianNormalizer(train_a_grady)
train_a_grady = agy_normalizer.encode(train_a_grady)
test_a_grady = agy_normalizer.encode(test_a_grady)

u_normalizer = UnitGaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)

train_theta = np.stack([train_a, train_a_smooth, train_a_gradx, train_a_grady], axis=-1)
test_theta = np.stack([test_a, test_a_smooth, test_a_gradx, test_a_grady], axis=-1)

print(train_a.shape)



##### construct multiple grid
# s=16
# level=3
level = int(np.log2(s) - 1)
print(level)
grid_list = []
theta_list = []
edge_index_list = []
edge_index_list_cuda = []


theta = train_theta
theta_d = 4
N = ntrain
K = s*s
for l in range(1, level+1):
    r_l = 2 ** (l - 1)
    s_l = s // r_l
    K_l = s_l * s_l
    print('level',s_l,r_l,K_l)
    xs = np.linspace(0.0, 1.0, s_l)
    ys = np.linspace(0.0, 1.0, s_l)
    grid_l = np.vstack([xx.ravel() for xx in np.meshgrid(xs, ys)]).T
    grid_l = torch.tensor(grid_l, dtype=torch.float)
    print(grid_l.shape)
    grid_list.append(grid_l)

    theta_l = theta.reshape(N, s, s, theta_d)
    theta_l = theta_l[:, ::r_l, ::r_l, :]
    theta_l = theta_l.reshape(N, K_l, theta_d)
    theta_l = torch.tensor(theta_l, dtype=torch.float)
    print(theta_l.shape)
    theta_list.append(theta_l)

    # for the finest level, we construct the nearest neighbors (NN)
    if l==1:
        edge_index_nn = []
        for x_i in range(s_l):
            for y_i in range(s_l):
                i = y_i * s_l + x_i
                for x in (-1,0,1):
                    for y in (-1,0,1):
                        x_j = x_i + x
                        y_j = y_i + y
                        j = y_j * s_l + x_j
                        # if (xj, yj) is a valid node
                        if (x_j in range(s_l)) and (y_j in range(s_l)) and (j!=i):
                            edge_index_nn.append([i,j])
        edge_index_nn = torch.tensor(edge_index_nn, dtype=torch.long)
        edge_index_nn = edge_index_nn.transpose(0,1)
        edge_index_list.append(edge_index_nn)
        edge_index_list_cuda.append(edge_index_nn.cuda())
        print(edge_index_nn.shape)

    # we then compute the interactive neighbors -- their parents are NN but they are net NN
    edge_index_inter = []
    for x_i in range(s_l):
        for y_i in range(s_l):
            i = y_i * s_l + x_i
            for x in range(-3,4):
                for y in range(-3,4):
                    x_j = x_i + x
                    y_j = y_i + y
                    j = y_j * s_l + x_j
                    # if (xj, yj) is a valid node
                    if (x_j in range(s_l)) and (y_j in range(s_l)):
                        # if (xi, yi), (xj, yj) not NN
                        if abs(x)>=2 or abs(y)>=2:
                            # if their parents are NN
                            if abs(x_i//2 - x_j//2)<=1 and abs(y_i//2 - y_j//2)<=1:
                                edge_index_inter.append([i,j])

    edge_index_inter = torch.tensor(edge_index_inter, dtype=torch.long)
    edge_index_inter = edge_index_inter.transpose(0,1)
    edge_index_list.append(edge_index_inter)
    edge_index_list_cuda.append(edge_index_inter.cuda())

print(len(grid_list),len(edge_index_list),len(theta_list))

def get_edge_attr(grid, theta, edge_index):
    n_edges = edge_index.shape[1]
    edge_attr = np.zeros((n_edges, 6))
    edge_attr[:, 0:4] = grid[edge_index.transpose(0,1)].reshape((n_edges, -1))
    edge_attr[:, 4] = theta[edge_index[0]]
    edge_attr[:, 5] = theta[edge_index[1]]
    return torch.tensor(edge_attr, dtype=torch.float)


data_train = []
for j in range(ntrain):
    X_list = []
    edge_attr_list = []
    for l in range(len(grid_list)):
        X_l = torch.cat([grid_list[l].reshape(-1, 2), theta_list[l][j].reshape(-1, 4)], dim=1).cuda()
        X_list.append(X_l)
    for i in range(len(edge_index_list)):
        if i==0:
            l = 0
        else:
            l = i-1
        edge_attr_l = get_edge_attr(grid_list[l], theta_list[l][j,:,0],  edge_index_list[i]).cuda()
        edge_attr_list.append(edge_attr_l)

    data_train.append((X_list, train_u[j].cuda(), edge_index_list_cuda, edge_attr_list))

# train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

model = KernelNN(width, ker_width, depth, edge_features, in_width=node_features).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
u_normalizer.cuda()
ttrain = np.zeros((epochs,))
ttest = np.zeros((epochs,))
model.train()
for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0.0
    # train_l2 = 0.0
    for data in data_train:
        X_list, y, edge_index_list, edge_attr_list = data

        optimizer.zero_grad()
        out = model(data)
        mse = F.mse_loss(out.view(-1, 1), y.view(-1, 1))
        mse.backward()

        # l2 = myloss(
        #     u_normalizer.decode(out.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)),
        #     u_normalizer.decode(batch.y.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)))
        optimizer.step()
        train_mse += mse.item()
        # train_l2 += l2.item()
    print(ep, train_mse / len(data_train))
    # scheduler.step()
    # t2 = default_timer()
    #
    # model.eval()
    # test_l2 = 0.0
    # with torch.no_grad():
    #     for batch in test_loader:
    #         batch = batch.to(device)
    #         out = model(batch)
    #         out = u_normalizer.decode(out.view(batch_size2, -1), sample_idx=batch.sample_idx.view(batch_size2, -1))
    #         test_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()
    #         # test_l2 += myloss(out.view(batch_size2,-1), y_normalizer.encode(batch.y.view(batch_size2, -1))).item()
    #
    # ttrain[ep] = train_l2 / (ntrain * k)
    # ttest[ep] = test_l2 / ntest

    # print(k, ntrain, ep, t2 - t1, train_mse / len(train_loader), train_l2 / (ntrain * k), test_l2 / ntest)

# np.savetxt(path_train_err, ttrain)
# np.savetxt(path_test_err, ttest)
# torch.save(model, path_model)
#
# plt.figure()
# # plt.plot(ttrain, label='train loss')
# plt.plot(ttest, label='test loss')
# plt.legend(loc='upper right')
# plt.show()
