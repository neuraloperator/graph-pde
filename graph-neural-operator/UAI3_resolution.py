import torch
import numpy as np

import torch.nn.functional as F

from torch_geometric.data import Data, DataLoader
from utilities import *
from nn_conv import NNConv_old

from timeit import default_timer


class KernelNN3(torch.nn.Module):
    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN3, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)

        kernel = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width_node, width_node, kernel, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)
        for k in range(self.depth):
            x = F.relu(self.conv1(x, edge_index, edge_attr))

        x = self.fc2(x)
        return x


TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'

for r in (1,2,4,8,16):

    # r = 2
    s = int(((241 - 1)/r) + 1)
    n = s**2
    m = 200
    k = 2

    radius_train = 0.25
    radius_test = 0.25
    print('resolution', s)


    ntrain = 100
    ntest = 100


    batch_size = 10
    batch_size2 = 10
    width = 64
    ker_width = 1000
    depth = 6
    edge_features = 6
    node_features = 6

    epochs = 200
    learning_rate = 0.0001
    scheduler_step = 50
    scheduler_gamma = 0.5


    path = 'UAI3_s'+str(s)
    path_model = 'model/' + path
    path_train_err = 'results/'+path+'train.txt'
    path_test_err1 = 'results/'+path+'test61.txt'
    path_test_err2 = 'results/'+path+'test121.txt'
    path_test_err3 = 'results/'+path+'test241.txt'

    t1 = default_timer()


    reader = MatReader(TRAIN_PATH)
    train_a = reader.read_field('coeff')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_smooth = reader.read_field('Kcoeff')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_gradx = reader.read_field('Kcoeff_x')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_a_grady = reader.read_field('Kcoeff_y')[:ntrain,::r,::r].reshape(ntrain,-1)
    train_u = reader.read_field('sol')[:ntrain,::r,::r].reshape(ntrain,-1)

    reader.load_file(TEST_PATH)
    test_a = reader.read_field('coeff')[:ntest,:,:]
    test_a_smooth = reader.read_field('Kcoeff')[:ntest,:,:]
    test_a_gradx = reader.read_field('Kcoeff_x')[:ntest,:,:]
    test_a_grady = reader.read_field('Kcoeff_y')[:ntest,:,:]
    test_u = reader.read_field('sol')[:ntest,:,:]

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


    test_a61 = test_a[:ntest, ::4, ::4].reshape(ntest, -1)
    test_a_smooth61 = test_a_smooth[:ntest, ::4, ::4].reshape(ntest, -1)
    test_a_gradx61 = test_a_gradx[:ntest, ::4, ::4].reshape(ntest, -1)
    test_a_grady61 = test_a_grady[:ntest, ::4, ::4].reshape(ntest, -1)
    test_u61 = test_u[:ntest, ::4, ::4].reshape(ntest, -1)

    test_a121 = test_a[:ntest, ::2, ::2].reshape(ntest, -1)
    test_a_smooth121 = test_a_smooth[:ntest, ::2, ::2].reshape(ntest, -1)
    test_a_gradx121 = test_a_gradx[:ntest, ::2, ::2].reshape(ntest, -1)
    test_a_grady121 = test_a_grady[:ntest, ::2, ::2].reshape(ntest, -1)
    test_u121 = test_u[:ntest, ::2, ::2].reshape(ntest, -1)

    test_a241 = test_a.reshape(ntest, -1)
    test_a_smooth241 = test_a_smooth.reshape(ntest, -1)
    test_a_gradx241 = test_a_gradx.reshape(ntest, -1)
    test_a_grady241 = test_a_grady.reshape(ntest, -1)
    test_u241 = test_u.reshape(ntest, -1)

    u_normalizer = GaussianNormalizer(train_u)
    train_u = u_normalizer.encode(train_u)
    # test_u = y_normalizer.encode(test_u)



    meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[s,s], sample_size=m)
    data_train = []
    for j in range(ntrain):
        for i in range(k):
            idx = meshgenerator.sample()
            grid = meshgenerator.get_grid()
            edge_index = meshgenerator.ball_connectivity(radius_train)
            edge_attr = meshgenerator.attributes(theta=train_a[j,:])
            #data_train.append(Data(x=init_point.clone().view(-1,1), y=train_y[j,:], edge_index=edge_index, edge_attr=edge_attr))
            data_train.append(Data(x=torch.cat([grid, train_a[j, idx].reshape(-1, 1),
                                                train_a_smooth[j, idx].reshape(-1, 1), train_a_gradx[j, idx].reshape(-1, 1),
                                                train_a_grady[j, idx].reshape(-1, 1)
                                                ], dim=1),
                                   y=train_u[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                                   ))


    meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[61,61], sample_size=m)
    data_test1 = []
    for j in range(ntest):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius_test)
        edge_attr = meshgenerator.attributes(theta=test_a61[j,:])
        data_test1.append(Data(x=torch.cat([grid, test_a61[j, idx].reshape(-1, 1),
                                           test_a_smooth61[j, idx].reshape(-1, 1), test_a_gradx61[j, idx].reshape(-1, 1),
                                           test_a_grady61[j, idx].reshape(-1, 1)
                                           ], dim=1),
                              y=test_u61[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                              ))
    #
    meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[121,121], sample_size=m)
    data_test2 = []
    for j in range(ntest):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius_test)
        edge_attr = meshgenerator.attributes(theta=test_a121[j,:])
        data_test2.append(Data(x=torch.cat([grid, test_a121[j, idx].reshape(-1, 1),
                                           test_a_smooth121[j, idx].reshape(-1, 1), test_a_gradx121[j, idx].reshape(-1, 1),
                                           test_a_grady121[j, idx].reshape(-1, 1)
                                           ], dim=1),
                              y=test_u121[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                              ))
    #
    meshgenerator = RandomMeshGenerator([[0,1],[0,1]],[241,241], sample_size=m)
    data_test3 = []
    for j in range(ntest):
        idx = meshgenerator.sample()
        grid = meshgenerator.get_grid()
        edge_index = meshgenerator.ball_connectivity(radius_test)
        edge_attr = meshgenerator.attributes(theta=test_a241[j,:])
        data_test3.append(Data(x=torch.cat([grid, test_a241[j, idx].reshape(-1, 1),
                                           test_a_smooth241[j, idx].reshape(-1, 1), test_a_gradx241[j, idx].reshape(-1, 1),
                                           test_a_grady241[j, idx].reshape(-1, 1)
                                           ], dim=1),
                              y=test_u241[j, idx], edge_index=edge_index, edge_attr=edge_attr, sample_idx=idx
                              ))
    #
    #
    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader1 = DataLoader(data_test1, batch_size=batch_size2, shuffle=False)
    test_loader2 = DataLoader(data_test2, batch_size=batch_size2, shuffle=False)
    test_loader3 = DataLoader(data_test3, batch_size=batch_size2, shuffle=False)


    t2 = default_timer()

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')

    model = KernelNN3(width, ker_width,depth,edge_features,in_width=node_features).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)
    u_normalizer.cuda()
    ttrain = np.zeros((epochs, ))
    ttest1 = np.zeros((epochs,))
    ttest2 = np.zeros((epochs,))
    ttest3 = np.zeros((epochs,))
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

            l2 = myloss(u_normalizer.decode(out.view(batch_size,-1), sample_idx=batch.sample_idx.view(batch_size,-1)),
                        u_normalizer.decode(batch.y.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size,-1)))

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        t2 = default_timer()

        model.eval()
        test1_l2 = 0.0
        test2_l2 = 0.0
        test3_l2 = 0.0

        with torch.no_grad():
            for batch in test_loader1:
                batch = batch.to(device)
                out = model(batch)
                out = u_normalizer.decode(out.view(batch_size2,-1), sample_idx=batch.sample_idx.view(batch_size2,-1))
                test1_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()
            for batch in test_loader2:
                batch = batch.to(device)
                out = model(batch)
                out = u_normalizer.decode(out.view(batch_size2,-1), sample_idx=batch.sample_idx.view(batch_size2,-1))
                test2_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()
            for batch in test_loader3:
                batch = batch.to(device)
                out = model(batch)
                out = u_normalizer.decode(out.view(batch_size2,-1), sample_idx=batch.sample_idx.view(batch_size2,-1))
                test3_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()


        ttrain[ep] = train_l2/(ntrain * k)
        ttest1[ep] = test1_l2 / ntest
        ttest2[ep] = test2_l2 / ntest
        ttest3[ep] = test3_l2 / ntest


        print(s, ep, t2-t1, train_mse/len(train_loader), train_l2/(ntrain * k))
        print(test1_l2/ntest, test2_l2/ntest, test3_l2/ntest)

    np.savetxt(path_train_err, ttrain)
    np.savetxt(path_test_err1, ttest1)
    np.savetxt(path_test_err2, ttest2)
    np.savetxt(path_test_err3, ttest3)
    torch.save(model, path_model)

