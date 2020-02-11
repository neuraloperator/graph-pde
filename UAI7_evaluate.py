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
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x


TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'

r = 2
s = int(((241 - 1)/r) + 1)
n = s**2
m = 200
k = 2
testm = 100

radius_train = 0.15
radius_test = 0.15

print('resolution', s)


ntrain = 100
ntest = 100

batch_size = 4
batch_size2 = 4 # must be a factor of num of samples
width = 64
ker_width = 1024
depth = 6
edge_features = 6
node_features = 6

epochs = 100
learning_rate = 0.0001
scheduler_step = 50
scheduler_gamma = 0.5


testr1 = 4
tests1 = int(((241 - 1)/r) + 1) ##61
test_split = 4 ##16
testn1 = s**2
path_train_err = 'results/UAI7_r'+str(s)+'train.txt'
path_test_err = 'results/UAI7_r'+str(s)+'_s'+ str(tests1)+'test.txt'
path_image = 'image/UAI7_r'+str(s)+'_s'+ str(tests1)


t1 = default_timer()


reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('coeff')[:ntest,::testr1,::testr1].reshape(ntest,-1)
test_a_smooth = reader.read_field('Kcoeff')[:ntest,::testr1,::testr1].reshape(ntest,-1)
test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::testr1,::testr1].reshape(ntest,-1)
test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::testr1,::testr1].reshape(ntest,-1)
test_u = reader.read_field('sol')[:ntest,::testr1,::testr1].reshape(ntest,-1)


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

u_normalizer = GaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)


meshgenerator = RandomMeshGenerator([[0, 1], [0, 1]], [s, s], sample_size=m)
data_train = []
for j in range(ntrain):
    for i in range(k):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius_train)
        edge_attr = meshgenerator.attributes(theta=train_a[j, :])
        data_train.append(Data(x=torch.cat([grid, train_a[j, idx].reshape(-1, 1),
                                            train_a_smooth[j, idx].reshape(-1, 1), train_a_gradx[j, idx].reshape(-1, 1),
                                            train_a_grady[j, idx].reshape(-1, 1)
                                            ], dim=1),
                               y=train_u[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                               ))


meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[tests1,tests1])
grid = meshgenerator.get_grid()
gridsplitter = LargeGridSplitter( grid, resolution=tests1, r=test_split, m=testm, radius=radius_test)

data_test = []
for j in range(ntest):
    theta = x=torch.cat([test_a[j,:].reshape(-1, 1),
                                       test_a_smooth[j,:].reshape(-1, 1), test_a_gradx[j,:].reshape(-1, 1), test_a_grady[j,:].reshape(-1, 1)
                                       ], dim=1)
    data_equation = gridsplitter.get_data(theta)
    equation_loader = DataLoader(data_equation, batch_size=batch_size2, shuffle=False)
    data_test.append(equation_loader)


print('grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
# print('edge_index_boundary', edge_index_boundary.shape, 'edge_attr', edge_attr_boundary.shape)

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)




##################################################################################################

### training

##################################################################################################
t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

model = KernelNN(width,ker_width,depth,edge_features,node_features).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
u_normalizer.cuda()
gridsplitter.cuda()

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
        # mse.backward()
        loss = torch.norm(out.view(-1) - batch.y.view(-1),1)
        loss.backward()

        l2 = myloss(out.view(batch_size,-1), batch.y.view(batch_size, -1))
        # l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for i, equation_loader in enumerate(data_test):
            pred = []
            for batch in equation_loader:
                batch = batch.to(device)
                out = model(batch)
                pred.append(out)

            out = gridsplitter.assemble(pred)
            y = test_u[i].to(device)
            test_l2 += myloss(u_normalizer.decode(out.view(1,-1)), y.view(1, -1)).item()

    ttrain[ep] = train_mse/len(train_loader)
    ttest[ep] = test_l2/ntest

    print(ep, t2-t1, train_mse/len(train_loader), train_l2/(ntrain), test_l2/ntest)

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

