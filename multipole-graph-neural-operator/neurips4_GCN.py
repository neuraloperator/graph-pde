import torch
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import Data, DataLoader
import matplotlib.pyplot as plt
from utilities import *
from torch_geometric.nn import GCNConv, NNConv

from timeit import default_timer
import scipy.io

torch.manual_seed(0)
np.random.seed(0)



class GCN_Net(torch.nn.Module):
    def __init__(self, width, ker_width, depth, in_width=1, out_width=1):
        super(GCN_Net, self).__init__()
        self.depth = depth
        self.width = width

        self.fc_in = torch.nn.Linear(in_width, width)

        self.conv1 = GCNConv(width, width)
        self.conv2 = GCNConv(width, width)
        self.conv3 = GCNConv(width, width)
        self.conv4 = GCNConv(width, width)


        self.fc_out1 = torch.nn.Linear(width, ker_width)
        self.fc_out2 = torch.nn.Linear(ker_width, 1)


    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.fc_in(data.x)

        for t in range(self.depth):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = self.conv2(x, edge_index)
            x = F.relu(x)
            x = self.conv3(x, edge_index)
            x = F.relu(x)
            x = self.conv4(x, edge_index)
            x = F.relu(x)

        x = F.relu(self.fc_out1(x))
        x = self.fc_out2(x)
        return x



TRAIN_PATH = 'data/piececonst_r421_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r421_N1024_smooth2.mat'


r = 1
s = int(((421 - 1)/r) + 1)
n = s**2
k = 1


print('resolution', s)

ntrain = 1024
ntest = 100

batch_size = 1
batch_size2 = 1

width = 128
ker_width = 1024
depth = 4

node_features = 6

epochs = 51
learning_rate = 0.0001
scheduler_step = 10
scheduler_gamma = 0.85



path = 'neurips4_GCN_s'+str(s)+'_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width)
path_model = 'model/' + path
path_train_err = 'results/' + path + 'train.txt'
path_test_err = 'results/' + path + 'test.txt'
path_image = 'results/' + path


t1 = default_timer()


reader = MatReader(TRAIN_PATH)
train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_a = reader.read_field('coeff')[:ntest,::r,::r].reshape(ntest,-1)
test_a_smooth = reader.read_field('Kcoeff')[:ntest,::r,::r].reshape(ntest,-1)
test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,::r,::r].reshape(ntest,-1)
test_a_grady = reader.read_field('Kcoeff_y')[:ntest,::r,::r].reshape(ntest,-1)
test_u = reader.read_field('sol')[:ntest,::r,::r].reshape(ntest,-1)


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
# test_u = y_normalizer.encode(test_u)

X, edge_index, _ = grid_edge(s, s)

data_train = []
for j in range(ntrain):
    for i in range(k):
        x = torch.cat([X, train_a[j].reshape(-1, 1),
                       train_a_smooth[j].reshape(-1, 1),
                       train_a_gradx[j].reshape(-1, 1),
                       train_a_grady[j].reshape(-1, 1)
                       ], dim=1)
        data_train.append(Data(x=x, y=train_u[j], edge_index=edge_index))

print(x.shape)
print(edge_index.shape)



data_test = []
for j in range(ntest):
    x = torch.cat([X, test_a[j].reshape(-1, 1),
                   test_a_smooth[j].reshape(-1, 1),
                   test_a_gradx[j].reshape(-1, 1),
                   test_a_grady[j].reshape(-1, 1)
                   ], dim=1)
    data_test.append(Data(x=x, y=test_u[j], edge_index=edge_index))

print(x.shape)
print(edge_index.shape)
train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(data_test, batch_size=batch_size2, shuffle=False)
t2 = default_timer()

print('preprocessing finished, time used:', t2-t1)
device = torch.device('cuda')

# print('use pre-train model')
# model = torch.load('model/multigraph_full_s141_ntrain1000_kerwidth1024')

model = GCN_Net(width=width, ker_width=ker_width, depth=depth, in_width=node_features,  out_width=1).cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

myloss = LpLoss(size_average=False)
ttrain = np.zeros((epochs, ))
ttest = np.zeros((epochs,))

for ep in range(epochs):
    t1 = default_timer()
    train_mse = 0.0
    train_l2 = 0.0
    model.train()
    u_normalizer.cuda()
    for batch in train_loader:
        batch = batch.to(device)

        optimizer.zero_grad()
        out = model(batch)
        mse = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
        # mse.backward()

        l2 = myloss(
            u_normalizer.decode(out.view(batch_size, -1)),
            u_normalizer.decode(batch.y.view(batch_size, -1)))
        l2.backward()

        optimizer.step()
        train_mse += mse.item()
        train_l2 += l2.item()

    scheduler.step()
    t2 = default_timer()
    ttrain[ep] = train_l2 / (ntrain * k)

    print(ep, t2 - t1, train_mse / len(train_loader), train_l2 / (ntrain * k))

    if ep % 10 == 0:
        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            t1 = default_timer()
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch)
                l2 = myloss(u_normalizer.decode(out.view(batch_size2, -1)), batch.y.view(batch_size2, -1))
                test_l2 += l2.item()


            ttest[ep] = test_l2 / ntest
            t2 = default_timer()
            print(ep, t2 - t1, test_l2 / (ntest*k) )
            # torch.save(model, path_model+str(ep))



np.savetxt(path_train_err, ttrain)
np.savetxt(path_test_err, ttest)


