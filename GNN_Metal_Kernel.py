import torch
import numpy as np
from torch_geometric.data import Data, DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import GCNConv
from nn_conv import NNConv, NNConv_Gaussian, NNConv_old
from utilities import *
from torch.autograd import Variable

from torch_geometric.data import InMemoryDataset
import matplotlib.pyplot as plt
import random

import pickle







random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

#======================================================#


Ntotal = 1024
num_train = 1024
num_test = 1024
num_data = num_train + num_test
num_data_per_batch = 64


depth = 4
width = 32

grid_size = 64
downsample_level = 4
resolution = grid_size // downsample_level
preprocess = False


nik = True
burigede = False

# TRAIN_PATH = 'data/Trainingdata2DfixedOrienLarge4Strain48000.mat'
TRAIN_PATH = 'data/piececonst_r65_N1024.mat'
TEST_PATH = 'data/piececonst_r65_N10000.mat'

path_preprocess = 'data/nik_1024_'+str(resolution)+'_aug'

path_train_err = 'results/MPaug_r'+str(resolution)+'_data10full_train.txt'
path_test_err = 'results/MPaug_r'+str(resolution)+'_data10full_test.txt'
path_image = 'image/MPaug_r'+str(resolution)+'_data10full.png'

epochs = 200



#################################################
#
# Network Architectures
#
#################################################


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3, width)
        self.conv1 = GCNConv(width, width)
        self.conv2 = GCNConv(width, width)
        self.conv3 = GCNConv(width, width)
        self.conv4 = GCNConv(width, width)
        self.conv5 = GCNConv(width, width)
        self.fc2 = nn.Linear(width, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.conv4(x, edge_index)
        x = F.relu(x)
        x = self.conv5(x, edge_index)
        x = F.relu(x)
        x = self.fc2(x)

        return x

class RNN_Net(torch.nn.Module):
    def __init__(self):
        super(RNN_Net, self).__init__()
        self.fc1 = nn.Linear(3, width)
        self.conv1 = GCNConv(width, width)
        self.fc2 = nn.Linear(width, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        for k in range(depth):
            x = self.conv1(x, edge_index)
            x = F.relu(x)

        x = self.fc2(x)
        return x

class RNN_Net3(torch.nn.Module):
    def __init__(self):
        super(RNN_Net3, self).__init__()
        self.fc1 = nn.Linear(3, width)
        self.conv1 = GCNConv(width, 2*width)
        self.conv2 = GCNConv(2*width, 2*width)
        self.conv3 = GCNConv(2*width, width)
        self.fc2 = nn.Linear(width, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.fc1(x)
        for k in range(depth):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)

        x = self.fc2(x)
        return x

class RNN_multi_grid(torch.nn.Module):
    def __init__(self):
        super(RNN_multi_grid, self).__init__()
        print('RNN_multi_grid')
        self.fc1 = nn.Linear(3, width)
        self.conv1 = GCNConv(width, width)
        self.fc2 = nn.Linear(width, 1)


    def forward(self, data):
        x, edge_index, mask_index = data.x, data.edge_index, data.mask_index

        x = self.fc1(x)

        for k in range(depth):
            x = self.conv1(x, edge_index)
            x = F.relu(x)

        x = x[mask_index]
        x = self.fc2(x)

        return x


class Net_MP(nn.Module):
    def __init__(self):
        super(Net_MP, self).__init__()
        self.fc1 = torch.nn.Linear(3, width)

        kernel = nn.Sequential(nn.Linear(3, width), nn.ReLU(),  nn.Linear(width, width**2))
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x

class Net_MP_diag(nn.Module):
    def __init__(self):
        super(Net_MP_diag, self).__init__()
        self.fc1 = torch.nn.Linear(3, width)

        kernel = nn.Sequential(nn.Linear(3, width//4), nn.ReLU(),  nn.Linear(width//4, width))
        self.conv1 = NNConv(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x

class Net_MP_diag3(nn.Module):
    def __init__(self):
        super(Net_MP_diag3, self).__init__()
        self.fc1 = torch.nn.Linear(3, width)

        kernel1 = nn.Sequential(nn.Linear(3+4, width//4), nn.ReLU(),  nn.Linear(width//4, width))
        self.conv1 = NNConv(width, width, kernel1, aggr='mean')
        kernel2 = nn.Sequential(nn.Linear(3+4, width // 4), nn.ReLU(), nn.Linear(width // 4, width))
        self.conv2 = NNConv(width, width, kernel2, aggr='mean')
        kernel3 = nn.Sequential(nn.Linear(3+4, width // 4), nn.ReLU(), nn.Linear(width // 4, width))
        self.conv3 = NNConv(width, width, kernel3, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            x = F.relu(self.conv2(x, edge_index, edge_attr))
            x = F.relu(self.conv3(x, edge_index, edge_attr))
        x = self.fc2(x)
        return x

class Net_MP_Gauss(nn.Module):
    def __init__(self):
        super(Net_MP_Gauss, self).__init__()
        self.fc1 = torch.nn.Linear(3, width)

        kernel = nn.Sequential(nn.Linear(3+4, width//4), nn.ReLU(),  nn.Linear(width//4, width))
        self.conv1 = NNConv(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x

class Net_MP_one(nn.Module):
    def __init__(self):
        super(Net_MP_one, self).__init__()

        kernel = nn.Sequential(nn.Linear(3, width), nn.ReLU(),  nn.Linear(width, 1))
        self.conv1 = NNConv_old(1, 1, kernel, aggr='mean')


    def forward(self, data):
        x, edge_index, edge_attr = data.x[:,2], data.edge_index, data.edge_attr

        for k in range(depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        return x

#======================================================================#


if(preprocess):
    print("generate graph")

    if(nik):
        traindata_loader = MatReader(TRAIN_PATH)
        traindata_input = traindata_loader.read_field('coeff').view(1024, 65, 65)[:num_train, 0:64, 0:64].reshape(num_train,-1)
        traindata_output = traindata_loader.read_field('sol').view(1024, 65, 65)[:num_train, 0:64, 0:64].reshape(num_train,-1)

        testdata_loader = MatReader(TEST_PATH)
        testdata_input = testdata_loader.read_field('coeff').view(10000, 65, 65)[:num_test, 0:64, 0:64].reshape(num_test,-1)
        testdata_output = testdata_loader.read_field('sol').view(10000, 65, 65)[:num_test, 0:64, 0:64].reshape(num_test,-1)

        traindata_input = downsample(traindata_input, grid_size, downsample_level)
        traindata_output = downsample(traindata_output, grid_size, downsample_level)
        testdata_input = downsample(testdata_input, grid_size, downsample_level)
        testdata_output = downsample(testdata_output, grid_size, downsample_level)
        grid_size = grid_size // downsample_level
        print(traindata_input.shape)
        print(traindata_output.shape)

        #### normalization
        x_normalizer = UnitGaussianNormalizer(traindata_input)
        x_train_enc = x_normalizer.encode(traindata_input)
        x_test_enc = x_normalizer.encode(testdata_input)
        data_input = torch.cat([x_train_enc, x_test_enc])

        y_normalizer = UnitGaussianNormalizer(traindata_output)
        y_train_enc = y_normalizer.encode(traindata_output)
        y_test_enc = y_normalizer.encode(testdata_output)
        data_output = torch.cat([y_train_enc, y_test_enc])


    if(burigede):

        #### read data

        data_loader = MatReader(TRAIN_PATH)
        data_input  = data_loader.read_field("theta_field").contiguous().view(Ntotal, -1)[:num_data, :]
        data_output = data_loader.read_field("Energyfield").contiguous().view(Ntotal, -1)[:num_data, :]


        #### down sample
        data_input = downsample(data_input, grid_size, downsample_level)
        data_output = downsample(data_output, grid_size, downsample_level)
        grid_size = grid_size // downsample_level
        print(data_input.shape)
        print(data_output.shape)

        #### normalization
        x_train = data_input[:num_train, :]
        x_test  = data_input[num_train:, :]
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train_enc = x_normalizer.encode(x_train)
        x_test_enc = x_normalizer.encode(x_test)
        data_input = torch.cat([x_train_enc,x_test_enc])

        y_train = data_output[:num_train, :]
        y_test  = data_output[num_train:, :]
        y_normalizer = UnitGaussianNormalizer(y_train)
        y_train_enc = y_normalizer.encode(y_train)
        y_test_enc = y_normalizer.encode(y_test)
        data_output = torch.cat([y_train_enc,y_test_enc])


    dataset = []
    for b in range(num_data):
        if b%100 == 0:
            print(b)
        theta = data_input[b, :]
        X, edge_index, edge_attr = grid_edge(grid_size, grid_size, theta)
        # X, edge_index, edge_attr, mask_index, num_nodes = multi_grid(depth=3, n_x=grid_size, n_y=grid_size, grid='grid_edge', params=theta)
        x = torch.tensor(X, dtype=torch.float)
        x = torch.cat([x,theta.reshape(-1,1)], dim=1)
        y = data_output[b,:]

        edge_index = torch.tensor(edge_index,dtype=torch.long)
        dataset.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr))
        # dataset.append(Data(x=x, y=y, edge_index=edge_index, edge_attr=edge_attr, mask_index=mask_index))

    pickle.dump(dataset, open(path_preprocess, "wb"))
    print('preprocessing finished')

    print(X.shape, edge_index.shape, edge_attr.shape)
else:
    dataset = pickle.load(open(path_preprocess, "rb"))

#==============================================================================#
# number of train data

# train_loader = DataLoader(dataset[:num_train], batch_size=num_data_per_batch, shuffle=True)
train_loader = DataLoader(dataset[:num_train], batch_size=num_data_per_batch, shuffle=True)
test_loader  = DataLoader(dataset[num_train:], batch_size=num_data_per_batch, shuffle=False)

#################################################
#
# train
#
#################################################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# model = RNN_Net().to(device)
model = RNN_Net3().to(device)
# model = RNN_multi_grid().to(device)
# model = Net_MP().to(device)
# model = Net_MP_diag().to(device)
# model = Net_MP_diag3().to(device)
# model = Net_MP_one().to(device)
# model = Net_MP_Gauss().to(device)
#
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)


loss_func = LpLoss(size_average=False)

test_loss = []
train_loss = []
model.train()
for epoch in range(epochs):
    train_error1 = 0
    train_error2 = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        #y = torch.cat([data.y for data in batch])
        loss1 = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
        loss2 = loss_func(out.view(-1, resolution ** 2), batch.y.view(-1, resolution ** 2)) / num_data_per_batch
        train_error1 = train_error1 + loss1
        train_error2 = train_error2 + loss2

        loss2.backward()
        optimizer.step()
        # scheduler.step()

    test_error = 0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            # test_error += F.mse_loss(pred.view(-1, 1), batch.y.view(-1, 1))
            test_error += loss_func(pred.view(-1, resolution ** 2), batch.y.view(-1, resolution ** 2)) / num_data_per_batch

    train_loss.append(train_error2 / len(train_loader))
    test_loss.append(test_error/len(test_loader) )
    # print(epoch, 'train loss: {:.4f}'.format(train_error/len(train_loader)),
                 # 'test L2 error: {:.4f}'.format(test_error/len(test_loader)))
    print(epoch, 'train loss1: {:.4f}'.format(train_error1/len(train_loader)),
                'train loss2: {:.4f}'.format(train_error2/len(train_loader)),
                 'test L2 error: {:.4f}'.format(test_error/len(test_loader)) )


np.savetxt(path_train_err, train_loss)
np.savetxt(path_test_err, test_loss)
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

truth = test_loader.dataset[0].y.detach().cpu().numpy().reshape((resolution, resolution))
model.cpu()
approx = model(test_loader.dataset[0]).detach().numpy().reshape((resolution, resolution))

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

plt.savefig(path_image)
