import torch
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import Data, DataLoader

from utilities import MatReader, UnitGaussianNormalizer, SquareMeshGenerator, DenseNet, LpLoss
from nn_conv import NNConv

from timeit import default_timer

class KernelNN(torch.nn.Module):
    def __init__(self, width, depth, ker_in, in_width=1, out_width=1):
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

t1 = default_timer()
TRAIN_PATH = 'Data/Darcy2d/piececonst_r121_N1024.mat'
TEST_PATH = 'Data/Darcy2d/piececonst_r121_N10000.mat'
INIT_PATH = 'Data/Darcy2d/poisson_r121_f1.mat'

ntrain = 1024
ntest = 100

reader = MatReader(TRAIN_PATH)
train_x = reader.read_field('coeff')[:ntrain,::2,::2].reshape(ntrain,-1)
train_y = reader.read_field('sol')[:ntrain,::2,::2].reshape(ntrain,-1)

reader.load_file(TEST_PATH)
test_x = reader.read_field('coeff')[:ntrain,::2,::2].reshape(ntrain,-1)
test_y = reader.read_field('sol')[:ntrain,::2,::2].reshape(ntrain,-1)

reader.load_file(INIT_PATH)
init_point = reader.read_field('sol')[::2,::2].reshape(-1)

s = 61

x_normalizer = UnitGaussianNormalizer(train_x)
train_x = x_normalizer.encode(train_x)
test_x = x_normalizer.encode(test_x)

y_normalizer = UnitGaussianNormalizer(train_y)
train_y = y_normalizer.encode(train_y)

meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[61,61])
edge_index = meshgenerator.ball_connectivity(0.05)

data_train = []
for j in range(ntrain):
	edge_attr = meshgenerator.attributes(theta=train_x[j,:])
	data_train.append(Data(x=init_point.clone().view(-1,1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))

data_test = []
for j in range(ntest):
	edge_attr = meshgenerator.attributes(theta=test_x[j,:])
	data_test.append(Data(x=init_point.clone().view(-1,1), y=test_y[j,:], edge_index=edge_index, edge_attr=edge_attr))

train_loader = DataLoader(data_train, batch_size=2, shuffle=True)
test_loader = DataLoader(data_test, batch_size=2, shuffle=False)

t2 = default_timer()

print(t2-t1)
device = torch.device('cuda')

model = KernelNN(32,4,5).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

myloss = LpLoss(size_average=False)
y_normalizer.cuda()

model.train()
for ep in range(100):
	t1 = default_timer()
	train_err = 0.0
	for batch in train_loader:
		batch = batch.to(device)

		optimizer.zero_grad()
		out = model(batch)
		loss = F.mse_loss(out.view(-1, 1), batch.y.view(-1,1))
		loss.backward()

		optimizer.step()
		train_err += loss.item()

	t2 = default_timer()

	test_err = 0.0
	with torch.no_grad():
		for batch in test_loader:
			batch = batch.to(device)

			out = model(batch).view(2,-1)
			test_err += myloss(y_normalizer.decode(out), batch.y.view(2,-1)).item()

	print(ep, t2-t1, test_err/ntest, train_err/ntrain)





#print(edge_index.size(), edge_attr.size(), t2-t1)
'''
edge_index = []
edge_attr = []

t1 = default_timer()
pwd = sklearn.metrics.pairwise_distances(meshgenerator.grid)
edge_index = np.vstack(np.where(pwd <= 0.5))

print(edge_index, edge_index.shape)
t2 = default_timer()
'''

'''
print(a.size, b.shape)

for j in range(meshgenerator.n):
	ind = np.where(pwd[j] <= 0.1)[0]
	for k in range(len(ind)):
		edge_index.append([j,ind[k]])

		attr = []
		attr.extend(meshgenerator.grid[j].tolist())
		attr.extend(meshgenerator.grid[k].tolist())

		edge_attr.append(attr)


#print(edge_attr)
print(len(edge_attr))
print(t2-t1)
'''

