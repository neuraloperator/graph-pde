import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from utilities import *
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import NNConv
from timeit import default_timer

torch.manual_seed(0)
np.random.seed(0)

########################################################################
#
#  The neural networks architecture
#
########################################################################


class MGKN(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, in_width, s):
        super(MGKN, self).__init__()
        self.depth = depth
        self.width = width
        self.s = s
        self.level = int(np.log2(s)-1)

        # P
        self.fc1 = torch.nn.Linear(in_width, width)

        # K_ll
        self.conv_list = []
        for l in range(self.level + 1):
            ker_width_l = max( ker_width // (2**l), 16)
            kernel_l = DenseNet([ker_in, ker_width_l, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_list.append(NNConv(width, width, kernel_l, aggr='mean'))
        self.conv_list = torch.nn.ModuleList(self.conv_list)

        # Q
        self.fc2 = torch.nn.Linear(width, ker_width)
        self.fc3 = torch.nn.Linear(ker_width, 1)


    # K_{l,l+1}
    def Upsample(self, x, channels, scale, s):
        x = x.transpose(0, 1).view(1,channels,s) # (K,width) to (1, width, s)
        x = F.upsample(x, scale_factor=scale, mode='nearest') # (1, width, s) to (1, width,  s*2)
        x = x.view(channels, -1).transpose(0, 1) # (1, width, s*2, s*2) to (K*4, width)
        return x

    # K_{l+1,l}
    def Downsample(self, x, channels, scale, s):
        x = x.transpose(0, 1).view(1,channels,s) # (K,width) to (1, width,  s)
        x = F.avg_pool1d(x, kernel_size=scale)
        x = x.view(channels, -1).transpose(0, 1) # (1, width, s/2, s/2) to (K/4, width)
        return x

    def forward(self, data):
        X_list,_, edge_index_list, edge_attr_list = data
        level = len(X_list)
        x = X_list[0]
        x = self.fc1(x)
        phi = [None] * level # list of x, len=level
        for k in range(self.depth):
            # downward
            for l in range(level):
                phi[l] = x
                if (l != level - 1):
                    # downsample
                    x = self.Downsample(x, channels=self.width, scale=2, s=self.s // (2 ** l) )

            # upward
            x = F.relu(x + self.conv_list[-1](phi[-1], edge_index_list[-1], edge_attr_list[-1]))
            for l in reversed(range(level)):
                if (l != 0):
                    # upsample
                    x = self.Upsample(x, channels=self.width, scale=2, s=self.s // (2 ** l))
                    # interactive neighbors
                    x = F.relu(x + self.conv_list[l](phi[l-1], edge_index_list[l], edge_attr_list[l]))
                else:
                    x = F.relu(x + self.conv_list[0](phi[0], edge_index_list[0], edge_attr_list[0]))

        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


########################################################################
#
#  Hyperparameters
#
########################################################################


r = 8 #downsample
s = 2**13//r #grid size after downsample

ntrain = 1024 # number of training pairs N
ntest = 100 # number of testing pairs

batch_size = 1 #train
batch_size2 = 1 #test
width = 64  #d_v
ker_width = 1024 #1024
depth = 4  #T
edge_features = 4
theta_d = 1
node_features = 1 + theta_d


epochs = 200
learning_rate = 0.00001
scheduler_step = 10
scheduler_gamma = 0.80


TRAIN_PATH = 'data/burgers_data_R10.mat'
TEST_PATH = 'data/burgers_data_R10.mat'

path = 'multipole_burgersR10_s'+str(s)+'_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width)
path_model = 'model/'+path
path_train_err = 'results/'+path+'train.txt'
path_test_err = 'results/'+path+'test.txt'
path_image = 'image/'+path



########################################################################
#
#  Read the data
#
########################################################################


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

train_theta = train_a.reshape(ntrain,s,1)
test_theta = test_a.reshape(ntest,s,1)


########################################################################
#
#  Construct Graphs
#
########################################################################



grid_list, train_theta_list, edge_index_list, edge_index_list_cuda = multi_pole_grid1d(theta = train_theta, theta_d=theta_d,s=s, N=ntrain, is_periodic=True)
grid_list, test_theta_list, edge_index_list, edge_index_list_cuda = multi_pole_grid1d(theta = test_theta, theta_d=theta_d,s=s, N=ntest, is_periodic=True)

data_train = []
for j in range(ntrain):
    X_list = []
    edge_attr_list = []
    for l in range(len(grid_list)):
        X_l = torch.cat([grid_list[l].reshape(-1, 1), train_theta_list[l][j].reshape(-1, theta_d)], dim=1).cuda()
        X_list.append(X_l)
    for i in range(len(edge_index_list)):
        if i==0:
            l = 0
        else:
            l = i-1
        edge_attr_l = get_edge_attr(grid_list[l], train_theta_list[l][j,:,0],  edge_index_list[i]).cuda()
        edge_attr_list.append(edge_attr_l)

    data_train.append((X_list, train_u[j].cuda(), edge_index_list_cuda, edge_attr_list))

# train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)

data_test = []
for j in range(ntest):
    X_list = []
    edge_attr_list = []
    for l in range(len(grid_list)):
        X_l = torch.cat([grid_list[l].reshape(-1, 1), test_theta_list[l][j].reshape(-1, theta_d)], dim=1).cuda()
        X_list.append(X_l)
    for i in range(len(edge_index_list)):
        if i==0:
            l = 0
        else:
            l = i-1
        edge_attr_l = get_edge_attr(grid_list[l], test_theta_list[l][j,:,0],  edge_index_list[i]).cuda()
        edge_attr_list.append(edge_attr_l)

    data_test.append((X_list, test_u[j].cuda(), edge_index_list_cuda, edge_attr_list))


########################################################################
#
#  Training
#
########################################################################

# print('use pre-train model')
# model = torch.load('model/multipole_burgersR10_s8192_ntrain1024_kerwidth1024')

model = MGKN(width, ker_width, depth, edge_features, in_width=node_features, s=s).cuda()



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
    train_l2 = 0.0
    for data in data_train:
        X_list, y, edge_index_list, edge_attr_list = data

        optimizer.zero_grad()
        out = model(data)
        mse = F.mse_loss(out.view(-1, 1), y.view(-1, 1))
        # mse.backward()
        l2_loss = myloss(u_normalizer.decode(out.view(1, -1)),
                           u_normalizer.decode(y.view(1, -1)))
        l2_loss.backward()
        train_l2 += l2_loss.item()

        optimizer.step()
        train_mse += mse.item()

    scheduler.step()
    t2 = default_timer()
    print(ep, t2 - t1, train_mse / len(data_train), train_l2 / len(data_train))
    ttrain[ep] = train_l2 / len(data_train)

torch.save(model, path_model)

########################################################################
#
#  Testing
#
########################################################################


model.eval()
test_l2 = 0.0
with torch.no_grad():
    t1 = default_timer()
    for i, data in enumerate(data_test):
        X_list, y, edge_index_list, edge_attr_list = data
        out = model(data)
        out = u_normalizer.decode(out.view(1, -1))
        loss =  myloss(out, y.view(1, -1)).item()
        test_l2 += loss
        print(i, loss)

        # resolution = s
        # coeff = test_a[i]
        # truth = y.detach().cpu().numpy()
        # approx = out.detach().cpu().numpy()
        #
        # np.savetxt('results/coeff'+str(i)+'.txt', coeff)
        # np.savetxt('results/truth' + str(i) + '.txt', truth)
        # np.savetxt('results/approx' + str(i) + '.txt', approx)


    t2 = default_timer()

print(epochs, t2 - t1,  test_l2 / ntest)
ttest[0] = test_l2 / ntest


np.savetxt(path_train_err, ttrain)
np.savetxt(path_test_err, ttest)

