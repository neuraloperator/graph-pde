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



class KernelNNBoundary(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width=1, out_width=1):
        super(KernelNNBoundary, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width)

        kernel = DenseNet([ker_in, ker_width//2, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')
        kernel2 = DenseNet([ker_in, ker_width//2, ker_width, width**2], torch.nn.ReLU)
        self.conv2 = NNConv_old(width, width, kernel2, aggr='mean')

        self.fc2 = torch.nn.Linear(width, ker_width)
        self.fc3 = torch.nn.Linear(ker_width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        edge_index_boundary, edge_attr_boundary = data.edge_index_boundary, data.edge_attr_boundary
        x = self.fc1(x)
        for k in range(self.depth):
            x = self.conv1(x, edge_index, edge_attr) + self.conv2(x, edge_index_boundary, edge_attr_boundary)
            x = F.relu(x) + self.fc1(data.x)

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

s0 = 241

TRAIN_PATH = 'data/grain2_s242_N1000_smooth.mat'
TEST_PATH = TRAIN_PATH

ntrain = 900
ntest = 100


batch_size = 20
batch_size2 = 20

r = 24
s = int(((s0 - 1)/r) + 1)
testr1 = r
tests1 = s
n = s**2

radius_train = 0.25
radius_test = 0.25
stride = 3
print('resolution', s)

epochs = 200
learning_rate = 0.001
scheduler_step = 20
scheduler_gamma = 0.5


width = 64
ker_width = 128
depth = 4
edge_features = 6
node_features = 7


path = 'grain_new_r'+str(s)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path



ttrain = np.zeros((epochs, ))
ttest = np.zeros((epochs,))

t1 = default_timer()


reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('theta')[:ntrain,:s0:r,:s0:r].reshape(ntrain,-1)
train_g = reader.read_field('grain')[:ntrain,:s0:r,:s0:r].reshape(ntrain,-1)
train_a_smooth = reader.read_field('Ktheta')[:ntrain,:s0:r,:s0:r].reshape(ntrain,-1)
train_a_gradx = reader.read_field('gradx')[:ntrain,:s0:r,:s0:r].reshape(ntrain,-1)
train_a_grady = reader.read_field('grady')[:ntrain,:s0:r,:s0:r].reshape(ntrain,-1)
train_u = reader.read_field('u')[:ntrain,:s0:r,:s0:r].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('theta')[-ntest:,:s0:testr1,:s0:testr1].reshape(ntest,-1)
test_g = reader.read_field('grain')[-ntest:,:s0:testr1,:s0:testr1].reshape(ntest,-1)
test_a_smooth = reader.read_field('Ktheta')[-ntest:,:s0:testr1,:s0:testr1].reshape(ntest,-1)
test_a_gradx = reader.read_field('gradx')[-ntest:,:s0:testr1,:s0:testr1].reshape(ntest,-1)
test_a_grady = reader.read_field('grady')[-ntest:,:s0:testr1,:s0:testr1].reshape(ntest,-1)
test_u = reader.read_field('u')[-ntest:,:s0:testr1,:s0:testr1].reshape(ntest,-1)


a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)
ag_normalizer = GaussianNormalizer(train_g)
train_g = ag_normalizer.encode(train_g)
test_g = ag_normalizer.encode(test_g)
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
# test_u = y_normalizer.encode(test_u)



meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[s,s])
edge_index = meshgenerator.ball_connectivity(radius_train)
grid = meshgenerator.get_grid()
meshgenerator.get_boundary()
edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)

data_train = []
for j in range(ntrain):
    edge_attr = meshgenerator.attributes(theta=train_a[j,:])
    edge_attr_boundary = meshgenerator.attributes_boundary(theta=train_a[j,:])
    data_train.append(Data(x= torch.cat([grid.reshape(-1, 2), train_a[j, :].reshape(-1, 1), train_g[j, :].reshape(-1, 1),
                               train_a_smooth[j, :].reshape(-1, 1), train_a_gradx[j, :].reshape(-1, 1),
                               train_a_grady[j, :].reshape(-1, 1)
                               ], dim=1),
                           y=train_u[j,:],
                           edge_index=edge_index, edge_attr=edge_attr,
                           edge_index_boundary=edge_index_boundary, edge_attr_boundary= edge_attr_boundary))


meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[s,s])
edge_index = meshgenerator.ball_connectivity(radius_test)
grid = meshgenerator.get_grid()
meshgenerator.get_boundary()
edge_index_boundary = meshgenerator.boundary_connectivity2d(stride = stride)
data_test = []
for j in range(ntest):
    edge_attr = meshgenerator.attributes(theta=test_a[j,:])
    edge_attr_boundary = meshgenerator.attributes_boundary(theta=test_a[j, :])
    data_test.append(Data(x=torch.cat([grid.reshape(-1, 2), test_a[j,:].reshape(-1, 1), test_g[j, :].reshape(-1, 1),
                                       test_a_smooth[j,:].reshape(-1, 1), test_a_gradx[j,:].reshape(-1, 1), test_a_grady[j,:].reshape(-1, 1)
                                       ], dim=1),
                          y=test_u[j, :],
                           edge_index=edge_index, edge_attr=edge_attr,
                           edge_index_boundary=edge_index_boundary, edge_attr_boundary=edge_attr_boundary))

print('grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
print('edge_index_boundary', edge_index_boundary.shape, 'edge_attr', edge_attr_boundary.shape)


train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)

t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

# model = KernelNN(width,depth,edge_features,in_width=3).cuda()
model = KernelNNBoundary(width,ker_width,depth,edge_features,in_width=node_features).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=True)
u_normalizer.cuda()

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
        loss = myloss(u_normalizer.decode(out.view(batch_size, -1)),
                      u_normalizer.decode(batch.y.view(batch_size, -1)))
        # loss.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += loss.item()

    scheduler.step()
    t2 = default_timer()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            out = model(batch)
            test_l2 += myloss(u_normalizer.decode(out.view(batch_size2,-1)),
                              batch.y.view(batch_size2, -1)).item()

    ttrain[ep] = train_l2/len(train_loader)
    ttest[ep] = test_l2/len(test_loader)

    print(ep, t2-t1, train_mse/len(train_loader), train_l2/len(train_loader), test_l2/len(test_loader))

np.savetxt(path_train_err, ttrain)
np.savetxt(path_test_err, ttest)

plt.figure()
plt.plot(ttrain, label='train loss')
plt.plot(ttest, label='test loss')
plt.legend(loc='upper right')
plt.show()


resolution = s
data = train_loader.dataset[1]
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
u_normalizer.cpu()
approx = u_normalizer.decode(model(data).view(1,-1)).detach().numpy().reshape((resolution, resolution))

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
plt.imshow(abs(approx - truth))
plt.xticks([], [])
plt.yticks([], [])
plt.colorbar(fraction=0.046, pad=0.04)
plt.title('Error')

plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()

plt.savefig(path_image + '_test.png')
