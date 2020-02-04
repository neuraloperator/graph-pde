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






r = 4
s = 16
n = s**2
m = 100
k = 1

radius_train = 0.2
radius_test = 0.2

print('resolution', s)


ntrain = 100
ntest = 100

batch_size = 1
batch_size2 = 1
width = 32
ker_width = 64
width_mid = 2
depth = 3
edge_features = 4
node_features = 3

epochs = 20
learning_rate = 0.001
scheduler_step = 50
scheduler_gamma = 0.8



path_train_err = 'results/encoder_r'+str(s)+'_n'+ str(ntrain)+'train.txt'
path_test_err = 'results/encoder_r'+str(s)+'_n'+ str(ntrain)+'test.txt'
path_image = 'image/encoder_r'+str(s)+'_n'+ str(ntrain)+''


t1 = default_timer()




Ntotal     = 48000
num_train = 100
num_test = 100
num_data_per_batch = 64
num_test = Ntotal-num_train

X_TRAIN_FIELD = 'Energyfield'
Y_TRAIN_FIELD = X_TRAIN_FIELD
grid_size = s  # Only for plotting
#data_loader = MatReader(TRAIN_PATH,True,True,True)

TRAIN_PATH = 'data/Trainingdata2DfixedOrienLarge4Strain48000.mat'
data_loader = MatReader(TRAIN_PATH)
data_input  = data_loader.read_field(X_TRAIN_FIELD).contiguous().view(Ntotal, -1)


x_train  = data_input[:num_train,:]
x_test   = data_input[num_test:,:]
x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

x_train16 = x_train.view(-1,64,64)[:,::4,::4]
x_train4 = x_train.view(-1,64,64)[:,::16,::16]
x_test16 = x_test.view(-1,64,64)[:,::4,::4]
x_test4 = x_test.view(-1,64,64)[:,::16,::16]


meshgenerator1 = SquareMeshGenerator([[0,1],[0,1]],[s,s])
meshgenerator2 = SquareMeshGenerator([[0,1],[0,1]],[s//4,s//4])
meshgenerator3 = SquareMeshGenerator([[0,1],[0,1]],[s,s])

grid1 = meshgenerator1.get_grid()
grid2 = meshgenerator2.get_grid()
grid3 = meshgenerator3.get_grid()
n_node1 = grid1.shape[0]
n_node2 = grid2.shape[0]
n_node3 = grid3.shape[0]

grid = torch.cat([grid1, grid2, grid3], dim=0)

edge_index_inner1 = meshgenerator1.ball_connectivity(radius_train)
edge_index_inner2 = meshgenerator2.ball_connectivity(radius_train) #+ n_node1
edge_index_inner3 = meshgenerator3.ball_connectivity(radius_train) #+ n_node1 + n_node2

edge_attr_inner1 = meshgenerator1.attributes()
edge_attr_inner2 = meshgenerator2.attributes()
edge_attr_inner3 = meshgenerator3.attributes()

pwd12 = sklearn.metrics.pairwise_distances(grid1, grid2)
edge_index_inter12 = np.vstack(np.where(pwd12 <= radius_train))
edge_index_inter12[:,1] += n_node1
edge_attr_inter12 = grid[edge_index_inter12.T].reshape((-1,2))

edge_index_inter21 = edge_index_inter12[[1,0],:]
edge_attr_inter21 = edge_attr_inter12[:,[1,0]]

pwd23 = sklearn.metrics.pairwise_distances(grid2, grid3)
edge_index_inter23 = np.vstack(np.where(pwd23 <= radius_train))
edge_index_inter23[:,0] += n_node1
edge_index_inter23[:,1] += n_node1 + n_node2
edge_attr_inter23 = grid[edge_index_inter23.T].reshape((-1,2))

edge_index_inter32 = edge_index_inter23[[1,0],:]
edge_attr_inter32 = edge_attr_inter23[:,[1,0]]

edge_index_inter12 = torch.tensor(edge_index_inter12, dtype=torch.long)
edge_index_inter21 = torch.tensor(edge_index_inter21, dtype=torch.long)
edge_index_inter23 = torch.tensor(edge_index_inter23, dtype=torch.long)
edge_index_inter32 = torch.tensor(edge_index_inter32, dtype=torch.long)

edge_index_inter2 = torch.cat([edge_index_inter12, edge_index_inter32], dim=1)
edge_attr_inter2 = torch.cat([edge_attr_inter12, edge_attr_inter32], dim=0)
edge_index_inter13 = torch.cat([edge_index_inter21, edge_index_inter23], dim=1)
edge_attr_inter13 = torch.cat([edge_attr_inter21, edge_attr_inter23], dim=0)

data_train = []
for j in range(ntrain):
    data_train.append(Data(x=torch.cat([grid[:n_node1], x_train16[j, :].reshape(-1, 1)], dim=1),
                           y=x_train16[j,:],
                           # edge_index_inner1=edge_index_inner1, edge_index_inner2=edge_index_inner2, edge_index_inner3=edge_index_inner3,
                           # edge_index_inter12=edge_index_inter12, edge_index_inter23=edge_index_inter23,
                           # edge_attr_inner1=edge_attr_inner1, edge_attr_inner2=edge_attr_inner2, edge_attr_inner3=edge_attr_inner3,
                           # edge_attr_inter12=edge_attr_inter12, edge_attr_inter23=edge_attr_inter23,
                           ))

data_test = []
for j in range(ntrain):
    # X = torch.cat([x_test16[j, :].reshape(-1, 1), x_test4[j, :].reshape(-1, 1), x_test16[j, :].reshape(-1, 1)], dim=0)
    data_test.append(Data(x=torch.cat([grid[:n_node1], x_test16[j, :].reshape(-1, 1)], dim=1),
                           y=x_test16[j, :],
                           # edge_index_inner1=edge_index_inner1, edge_index_inner2=edge_index_inner2, edge_index_inner3=edge_index_inner3,
                           # edge_index_inter12=edge_index_inter12, edge_index_inter23=edge_index_inter23,
                           # edge_attr_inner1=edge_attr_inner1, edge_attr_inner2=edge_attr_inner2, edge_attr_inner3=edge_attr_inner3,
                           # edge_attr_inter12=edge_attr_inter12, edge_attr_inter23=edge_attr_inter23,
                           ))

print('grid', grid.shape)
print('edge_index_inner1', edge_index_inner1.shape, 'edge_index_inner2', edge_index_inner2.shape, 'edge_index_inner3', edge_index_inner3.shape)
print('edge_attr_inner1', edge_attr_inner1.shape, 'edge_attr_inner2', edge_attr_inner2.shape, 'edge_attr_inner3', edge_attr_inner3.shape)
print('edge_index_inter12', edge_index_inter12.shape, 'edge_index_inter23', edge_index_inter23.shape)
print('edge_attr_inter12', edge_attr_inter12.shape, 'edge_attr_inter23', edge_attr_inter23.shape)

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

edge_index_inner1 = edge_index_inner1.cuda()
edge_index_inner2 = edge_index_inner2.cuda()
edge_index_inner3 = edge_index_inner3.cuda()
edge_attr_inner1 = edge_attr_inner1.cuda()
edge_attr_inner2 = edge_attr_inner2.cuda()
edge_attr_inner3 = edge_attr_inner3.cuda()
edge_index_inter12 = edge_index_inter12.cuda()
edge_index_inter21 = edge_index_inter21.cuda()
edge_index_inter23 = edge_index_inter23.cuda()
edge_index_inter32 = edge_index_inter32.cuda()
edge_attr_inter12 = edge_attr_inter12.cuda()
edge_attr_inter21 = edge_attr_inter21.cuda()
edge_attr_inter23 = edge_attr_inter23.cuda()
edge_attr_inter32 = edge_attr_inter32.cuda()

class KernelNN(torch.nn.Module):
    def __init__(self, width, width_mid, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel1 = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel1, aggr='mean')
        kernel2 = DenseNet([ker_in, ker_width, ker_width, width_mid**2], torch.nn.ReLU)
        self.conv2 = NNConv_old(width_mid, width_mid, kernel2, aggr='mean')
        kernel3 = DenseNet([ker_in, ker_width, ker_width, width**2], torch.nn.ReLU)
        self.conv3 = NNConv_old(width, width, kernel3, aggr='mean')

        kernel12 = DenseNet([ker_in, ker_width, ker_width, width * width_mid], torch.nn.ReLU)
        self.conv12 = NNConv_old(width, width_mid, kernel12, aggr='mean')

        kernel23 = DenseNet([ker_in, ker_width, ker_width, width * width_mid], torch.nn.ReLU)
        self.conv23 = NNConv_old(width_mid, width, kernel23, aggr='mean')

        kernel32 = DenseNet([ker_in, ker_width, ker_width, width * width_mid], torch.nn.ReLU)
        self.conv32 = NNConv_old(width, width_mid, kernel32, aggr='mean')

        kernel21 = DenseNet([ker_in, ker_width, ker_width, width * width_mid], torch.nn.ReLU)
        self.conv21 = NNConv_old(width_mid, width, kernel21, aggr='mean')

        self.fc2 = torch.nn.Linear(width, 1)

    def forward(self, data):
        x = data.x
        x1 = self.fc1(x)
        x2 = torch.zeros(n_node2, width_mid).cuda()
        x3 = torch.zeros(n_node3, width).cuda()
        x_wide = torch.zeros(n_node1 + n_node2 + n_node3, width).cuda()
        x_narrow = torch.zeros(n_node1 + n_node2 + n_node3, width_mid).cuda()

        for k in range(self.depth):
            x1 = F.relu(self.conv1(x1, edge_index_inner1, edge_attr_inner1))
            x_wide[:n_node1,:] = x1

            x2 = x2 + F.relu(self.conv12(x_wide, edge_index_inter12, edge_attr_inter12))[n_node1:n_node1+n_node2,:]
            x2 = F.relu(self.conv2(x2, edge_index_inner2, edge_attr_inner2))
            x_narrow[n_node1:n_node1+n_node2, :] = x2

            x3 = x3 + F.relu(self.conv23(x_narrow, edge_index_inter23, edge_attr_inter23))[n_node1+n_node2:,:]
            x3 = F.relu(self.conv3(x3, edge_index_inner3, edge_attr_inner3))
            x_wide[n_node1 + n_node2:, :] = x3

            x2 = x2 + F.relu(self.conv32(x_wide, edge_index_inter32, edge_attr_inter32))[n_node1:n_node1+n_node2,:]
            x2 = F.relu(self.conv2(x2, edge_index_inner2, edge_attr_inner2))
            x_narrow[n_node1:n_node1 + n_node2, :] = x2

            x1 = x1 + F.relu(self.conv21(x_narrow, edge_index_inter21, edge_attr_inter21))[:n_node1,:]

        x = self.fc2(x3)
        return x


model = KernelNN(width=width, width_mid=width_mid, ker_width=ker_width, depth=depth, ker_in=edge_features, in_width=node_features).cuda()


##################################################################################################

### training

##################################################################################################
t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')


optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)


model.train()
ttrain = np.zeros((epochs, ))
ttest = np.zeros((epochs,))
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

        # l2 = myloss(out.view(batch_size,-1), batch.y.view(batch_size, -1))

        optimizer.step()
        train_mse += mse.item()
        # train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()

    model.eval()
    test_mse = 0.0
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            test_mse += F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
            # test_l2 += myloss(u_normalizer.decode(out.view(batch_size2,-1)), batch.y.view(batch_size2, -1)).item()
            # test_l2 += myloss(out.view(batch_size2,-1), y_normalizer.encode(batch.y.view(batch_size2, -1))).item()

    ttrain[ep] = train_mse/len(train_loader)
    ttest[ep] = test_mse/len(test_loader)

    print(ep, t2-t1, train_mse/len(train_loader), test_mse/len(test_loader))

np.savetxt(path_train_err, ttrain)
np.savetxt(path_test_err, ttest)


##################################################################################################

### Ploting

##################################################################################################


plt.figure()
# plt.plot(ttrain, label='train loss')
plt.plot(ttest, label='test loss')
plt.legend(loc='upper right')
plt.show()


resolution = s
data = train_loader.dataset[0]
truth = data.y.detach().cpu().numpy().reshape((resolution, resolution))
model.cpu()
approx = model(data).detach().numpy().reshape((resolution, resolution))

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
# plt.show()

plt.savefig(path_image + '_train.png')



data = test_loader.dataset[0]
truth = data.y.detach().cpu().numpy().reshape((resolution, resolution))
model.cpu()
approx = model(data).detach().numpy().reshape((resolution, resolution))

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
plt.show()

plt.savefig(path_image + '_test.png')
