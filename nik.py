import torch
import numpy as np

import torch.nn.functional as F
import torch.nn as nn

from torch_geometric.data import Data, DataLoader

from utilities import MatReader, UnitGaussianNormalizer, SquareMeshGenerator, DenseNet, LpLoss
from nn_conv import NNConv

from timeit import default_timer
import scipy.io

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


TRAIN_PATH = 'Data/Darcy2d/piececonst_r121_N1024.mat'
TEST_PATH = 'Data/Darcy2d/piececonst_r121_N10000.mat'
INIT_PATH = 'Data/Darcy2d/poisson_r121_f1.mat'

ntrain = 1024
ntest = 1024

ttrain = np.zeros((5,500))
ttest = np.zeros((5,))
examp = []
for m in range(5):
	t1 = default_timer()
	r = m + 2

	reader = MatReader(TRAIN_PATH)
	train_x = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
	train_y = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

	reader.load_file(TEST_PATH)
	test_x = reader.read_field('coeff')[:ntest,::r,::r].reshape(ntest,-1)
	test_y = reader.read_field('sol')[:ntest,::r,::r].reshape(ntest,-1)

	reader.load_file(INIT_PATH)
	init_point = reader.read_field('sol')[::r,::r].reshape(-1)

	s = int(((121 - 1)/r) + 1)
	print(s)

	x_normalizer = UnitGaussianNormalizer(train_x)
	train_x = x_normalizer.encode(train_x)
	test_x = x_normalizer.encode(test_x)

	y_normalizer = UnitGaussianNormalizer(train_y)
	train_y = y_normalizer.encode(train_y)

	meshgenerator = SquareMeshGenerator([[0,1],[0,1]],[s,s])
	edge_index = meshgenerator.ball_connectivity(0.05)
	grid = meshgenerator.get_grid()

	data_train = []
	for j in range(ntrain):
		edge_attr = meshgenerator.attributes(theta=train_x[j,:])
		#data_train.append(Data(x=init_point.clone().view(-1,1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
		data_train.append(Data(x=torch.cat([grid, train_x[j,:].reshape(-1,1)], dim=1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))

	data_test = []
	for j in range(ntest):
		edge_attr = meshgenerator.attributes(theta=test_x[j,:])
		#data_test.append(Data(x=init_point.clone().view(-1,1), y=test_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
		data_test.append(Data(x=torch.cat([grid, test_x[j,:].reshape(-1,1)], dim=1), y=test_y[j,:], edge_index=edge_index, edge_attr=edge_attr))

	train_loader = DataLoader(data_train, batch_size=2, shuffle=True)
	test_loader = DataLoader(data_test, batch_size=2, shuffle=False)

	t2 = default_timer()

	print(t2-t1)
	device = torch.device('cuda')

	model = KernelNN(32,4,5,in_width=3).cuda()
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

	model.train()
	for ep in range(500):
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

		scheduler.step()

		t2 = default_timer()

		ttrain[m,ep] = train_err/ntrain

		print(ep, t2-t1, ttrain[m,ep])

	myloss = LpLoss(size_average=False)
	y_normalizer.cuda()
	model.eval()
	test_err = 0.0
	with torch.no_grad():
		for k, batch in enumerate(test_loader):
			batch = batch.to(device)

			out = model(batch).view(2,-1)
			test_err += myloss(y_normalizer.decode(out), batch.y.view(2,-1)).item()

			if k == 0:
				examp.append(y_normalizer.decode(out)[0,:].view(s,s).cpu().numpy())

	ttest[m] = test_err/ntest

	print(ttest[m])

scipy.io.savemat('graph_res.mat', mdict={'train': ttrain, 'test': ttest, 'ex1': examp[0], 'ex2': examp[1], 'ex3': examp[2], 'ex4': examp[3], 'ex5': examp[4]})


