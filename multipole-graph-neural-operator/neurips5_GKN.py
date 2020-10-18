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

        kernel = DenseNet([ker_in, ker_width//2, ker_width, width**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width, width, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width, ker_width)
        self.fc3 = torch.nn.Linear(ker_width, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = self.conv1(x, edge_index, edge_attr)
            if k != self.depth-1:
                x = F.relu(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        return x

# torch.cuda.set_device('cuda:3')
TRAIN_PATH = 'data/burgers_data_R10.mat'
TEST_PATH = 'data/burgers_data_R10.mat'
# TRAIN_PATH = 'data/burgers1d_small.mat'
# TEST_PATH = 'data/burgers1d_small.mat'

r = 8
s = 2**13//r
K = s

ntrain = 32
ntest = 32

n = s
k = 2
trainm = 128
assert n % trainm == 0
train_split = s // trainm

testr1 = r
tests1 = 2**13 // testr1
test_split = train_split
testn1 = s
testm = trainm

batch_size = 4 # factor of ntrain * k
batch_size2 = 4 # factor of test_split

radius_train = 0.20
radius_test = 0.20
# rbf_sigma = 0.2

print('resolution', s)




assert test_split%batch_size2 == 0 # the batchsize must divide the split

width = 64
ker_width = 1024
depth = 6
edge_features = 4
node_features = 2

epochs = 101
learning_rate = 0.0001
scheduler_step = 10
scheduler_gamma = 0.85


path = 'neurips5_GKN_r'+str(s)+'_s'+ str(tests1)+'testm'+str(testm)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path


t1 = default_timer()


reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('a')[:ntrain,::r].reshape(ntrain,-1)
train_u = reader.read_field('u')[:ntrain,::r].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('a')[-ntest:,::r].reshape(ntest,-1)
test_u = reader.read_field('u')[-ntest:,::r].reshape(ntest,-1)


a_normalizer = GaussianNormalizer(train_a)
train_a = a_normalizer.encode(train_a)
test_a = a_normalizer.encode(test_a)

u_normalizer = UnitGaussianNormalizer(train_u)
train_u = u_normalizer.encode(train_u)
# test_u = y_normalizer.encode(test_u)


meshgenerator = RandomMeshGenerator([[0,1]],[s], sample_size=trainm)
data_train = []
for j in range(ntrain):
    for i in range(k):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius_train)
        edge_attr = meshgenerator.attributes(theta=train_a[j, :])
        # data_train.append(Data(x=init_point.clone().view(-1,1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
        data_train.append(Data(x=torch.cat([grid.reshape(-1, 1), train_a[j, idx].reshape(-1, 1)], dim=1),
                               y=train_u[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                               ))

train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
# print('grid', grid.shape, 'edge_index', edge_index.shape, 'edge_attr', edge_attr.shape)
# print('edge_index_boundary', edge_index_boundary.shape, 'edge_attr', edge_attr_boundary.shape)


meshgenerator = SquareMeshGenerator([[0,1]],[tests1])
grid = meshgenerator.get_grid()
gridsplitter = RandomGridSplitter(grid, resolution=tests1, d=1, l=1, m=testm, radius=radius_test)

data_test = []
for j in range(ntest):
    theta =test_a[j,:].reshape(-1, 1)
    data_equation = gridsplitter.get_data(theta)
    equation_loader = DataLoader(data_equation, batch_size=batch_size2, shuffle=False)
    data_test.append(equation_loader)




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

# gridsplitter.cuda()


ttrain = np.zeros((epochs, ))
ttest = np.zeros((epochs,))
for ep in range(epochs):
    u_normalizer.cuda()
    model.train()
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

        l2 = myloss(
            u_normalizer.decode(out.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)),
            u_normalizer.decode(batch.y.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)))
        # l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    ttrain[ep] = train_l2 / (ntrain * k)
    scheduler.step()
    t2 = default_timer()


    print(ep, t2-t1, train_mse/len(train_loader), train_l2 / (ntrain * k))

    if ep % 20 == 0:
        model.eval()
        test_l2 = 0.0
        test_l2_split = 0.0
        u_normalizer.cpu()
        with torch.no_grad():
            for i, equation_loader in enumerate(data_test):
                pred = []
                split_idx = []
                for batch in equation_loader:
                    batch = batch.to(device)
                    out = model(batch).detach().cpu()
                    pred.append(out)
                    split_idx.append(batch.split_idx)

                    out_split = u_normalizer.decode(out.view(batch_size2, -1),
                                                    sample_idx=batch.split_idx.view(batch_size2, -1))
                    test_l2_split += myloss(out_split, test_u[i, batch.split_idx]).item()


                out = gridsplitter.assemble(pred, split_idx, batch_size2, sigma=1)
                out = u_normalizer.decode(out.view(1, -1))
                test_l2 += myloss(out, test_u[i].view(1, -1)).item()

            ttest[ep] = test_l2 / ntest
            t3 = default_timer()
            print(ep, t3-t2, test_l2/ntest, test_l2_split/(ntest*test_split))





np.savetxt(path_train_err, ttrain)
np.savetxt(path_test_err, ttest)
torch.save(model, path_model)
##################################################################################################

### Ploting

##################################################################################################



plt.figure()
# plt.plot(ttrain, label='train loss')
plt.plot(ttest, label='test loss')
plt.legend(loc='upper right')
plt.show()




