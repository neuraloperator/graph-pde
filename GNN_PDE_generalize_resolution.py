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


Ntotal = 2000
num_train = 10
num_test = 10
num_data = num_train + num_test
num_data_per_batch = 1


depth = 4
width = 32

epochs = 1000
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.1

resolution = 16
preprocess = False


nik = True
burigede = False

# TRAIN_PATH = 'data/Trainingdata2DfixedOrienLarge4Strain48000.mat'
TRAIN_PATH = 'data/piececonst_r65_N1024.mat'
TEST_PATH = 'data/piececonst_r65_N10000.mat'

path_preprocess_train = 'data/nik_r'+str(resolution)+'_10_full'
path_preprocess1 = 'data/nik_r'+str(16)+'_10_full'
path_preprocess2 = 'data/nik_r'+str(32)+'_10_full'
path_preprocess3 = 'data/nik_r'+str(64)+'_10_full'

path_train_err = 'results/MPaug_r'+str(resolution)+'_data10full_train_generalize.txt'
path_test_err = 'results/MPaug_r'+str(resolution)+'_data10full_test_generalize'
path_image = 'image/MPaug_r'+str(resolution)+'_data10full_generalize'





#################################################
#
# Network Architectures
#
#################################################



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



class Net_MP_diag3(nn.Module):
    def __init__(self):
        super(Net_MP_diag3, self).__init__()
        self.fc1 = torch.nn.Linear(3, width)

        kernel1 = nn.Sequential(nn.Linear(3, width//4), nn.ReLU(),  nn.Linear(width//4, width))
        self.conv1 = NNConv(width, width, kernel1, aggr='mean')
        kernel2 = nn.Sequential(nn.Linear(3, width // 4), nn.ReLU(), nn.Linear(width // 4, width))
        self.conv2 = NNConv(width, width, kernel2, aggr='mean')
        kernel3 = nn.Sequential(nn.Linear(3, width // 4), nn.ReLU(), nn.Linear(width // 4, width))
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


#======================================================================#


dataset0 = pickle.load(open(path_preprocess_train, "rb"))
dataset1 = pickle.load(open(path_preprocess1, "rb"))
dataset2 = pickle.load(open(path_preprocess2, "rb"))
dataset3 = pickle.load(open(path_preprocess3, "rb"))


#==============================================================================#
# number of train data

train_loader = DataLoader(dataset0[:num_train], batch_size=num_data_per_batch, shuffle=True)
test_loader1  = DataLoader(dataset1[num_train : num_train+num_test], batch_size=num_data_per_batch, shuffle=False)
test_loader2  = DataLoader(dataset2[num_train : num_train+num_test], batch_size=num_data_per_batch, shuffle=False)
test_loader3  = DataLoader(dataset3[num_train : num_train+num_test], batch_size=num_data_per_batch, shuffle=False)


#################################################
#
# train
#
#################################################



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = Net().to(device)
# model = RNN_Net().to(device)
# model = RNN_Net3().to(device)
# model = RNN_multi_grid().to(device)
# model = Net_MP().to(device)
# model = Net_MP_diag().to(device)
# model = Net_MP_diag3().to(device)
# model = Net_MP_one().to(device)
model = Net_MP_Gauss().to(device)
#
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs, 1e-6)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
loss_func = LpLoss(size_average=False)


train_loss = []
test_loss1 = []
test_loss2 = []
test_loss3 = []
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

    test_error1 = 0
    test_error2 = 0
    test_error3 = 0
    with torch.no_grad():
        for batch in test_loader1:
            batch = batch.to(device)
            pred = model(batch)
            # test_error += F.mse_loss(pred.view(-1, 1), batch.y.view(-1, 1))
            test_error1 += loss_func(pred.view(-1, 16 ** 2),
                                    batch.y.view(-1, 16 ** 2)) / num_data_per_batch
        for batch in test_loader2:
            batch = batch.to(device)
            pred = model(batch)
            # test_error += F.mse_loss(pred.view(-1, 1), batch.y.view(-1, 1))
            test_error2 += loss_func(pred.view(-1, 32 ** 2),
                                    batch.y.view(-1, 32 ** 2)) / num_data_per_batch
        for batch in test_loader3:
            batch = batch.to(device)
            pred = model(batch)
            # test_error += F.mse_loss(pred.view(-1, 1), batch.y.view(-1, 1))
            test_error3 += loss_func(pred.view(-1, 64 ** 2),
                                    batch.y.view(-1, 64 ** 2)) / num_data_per_batch


    train_loss.append(train_error2 / len(train_loader))
    test_loss1.append(test_error1 / len(test_loader1) )
    test_loss2.append(test_error2 / len(test_loader2))
    test_loss3.append(test_error3 / len(test_loader3))
    # print(epoch, 'train loss: {:.4f}'.format(train_error/len(train_loader)),
                 # 'test L2 error: {:.4f}'.format(test_error/len(test_loader)))
    print(epoch, 'train loss1: {:.4f}'.format(train_error1/len(train_loader)),
                'train loss2: {:.4f}'.format(train_error2/len(train_loader)),
                 'r16 test L2 error: {:.4f}'.format(test_error1/len(test_loader1)),
                'r32 test L2 error: {:.4f}'.format(test_error2 / len(test_loader2)),
                'r64 test L2 error: {:.4f}'.format(test_error3 / len(test_loader3))  )


# train_loss = train_loss.numpy()
# test_loss = test_loss.numpy()

np.savetxt(path_train_err, train_loss)
np.savetxt(path_test_err + '16.txt', test_loss1)
np.savetxt(path_test_err + '32.txt', test_loss2)
np.savetxt(path_test_err + '64.txt', test_loss3)
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



plt.figure()




r = np.random.randint(num_data_per_batch)

data = test_loader1.dataset[0]
truth = data.y.detach().cpu().numpy().reshape((16, 16))
model.cpu()
approx = model(data).detach().numpy().reshape((16, 16))

# plt.figure()
plt.subplot(3, 3, 1)
plt.imshow(truth)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Ground Truth')

plt.subplot(3, 3, 2)
plt.imshow(approx)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Approximation')

plt.subplot(3, 3, 3)
plt.imshow((approx - truth) ** 2)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Error')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
# plt.show()



data = test_loader2.dataset[0]
truth = data.y.detach().cpu().numpy().reshape((32, 32))
model.cpu()
approx = model(data).detach().numpy().reshape((32, 32))

# plt.figure()
plt.subplot(3, 3, 4)
plt.imshow(truth)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Ground Truth')

plt.subplot(3, 3, 5)
plt.imshow(approx)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Approximation')

plt.subplot(3, 3, 6)
plt.imshow((approx - truth) ** 2)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Error')

plt.subplots_adjust(wspace=0.5, hspace=0.5)



data = test_loader3.dataset[0]
truth = data.y.detach().cpu().numpy().reshape((64, 64))
model.cpu()
approx = model(data).detach().numpy().reshape((64, 64))

# plt.figure()
plt.subplot(3, 3, 7)
plt.imshow(truth)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Ground Truth')

plt.subplot(3, 3, 8)
plt.imshow(approx)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Approximation')

plt.subplot(3, 3, 9)
plt.imshow((approx - truth) ** 2)
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Error')

plt.subplots_adjust(wspace=0.5, hspace=0.5)




plt.savefig(path_image + '_test.png')
plt.show()

