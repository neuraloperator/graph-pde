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



class KernelInduced(torch.nn.Module):
    def __init__(self, width, ker_width, depth, ker_in, points, level, in_width=1, out_width=1):
        super(KernelInduced, self).__init__()
        self.depth = depth
        self.width = width
        self.level = level
        self.points = points
        self.points_total = np.sum(points)

        # in
        self.fc_in = torch.nn.Linear(in_width, width)
        # self.fc_in_list = []
        # for l in range(level):
        #     self.fc_in_list.append(torch.nn.Linear(in_width, width))
        # self.fc_in_list = torch.nn.ModuleList(self.fc_in_list)

        # K12 K23 K34 ...
        self.conv_down_list = []
        for l in range(1, level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_down_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_down_list = torch.nn.ModuleList(self.conv_down_list)

        # K11 K22 K33
        self.conv_list = []
        for l in range(level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_list = torch.nn.ModuleList(self.conv_list)

        # K21 K32 K43
        self.conv_up_list = []
        for l in range(1, level):
            ker_width_l = ker_width // (2 ** l)
            kernel_l = DenseNet([ker_in, ker_width_l, width ** 2], torch.nn.ReLU)
            self.conv_up_list.append(NNConv(width, width, kernel_l, aggr='mean', root_weight=False, bias=False))
        self.conv_up_list = torch.nn.ModuleList(self.conv_up_list)

        # out
        self.fc_out1 = torch.nn.Linear(width, ker_width)
        self.fc_out2 = torch.nn.Linear(ker_width, 1)


    def forward(self, data):
        edge_index_down, edge_attr_down, range_down = data.edge_index_down, data.edge_attr_down, data.edge_index_down_range
        edge_index_mid, edge_attr_mid, range_mid = data.edge_index_mid, data.edge_attr_mid, data.edge_index_range
        edge_index_up, edge_attr_up, range_up = data.edge_index_up, data.edge_attr_up, data.edge_index_up_range

        x = self.fc_in(data.x)

        for t in range(self.depth):
            #downward
            for l in range(self.level-1):
                x = x + self.conv_down_list[l](x, edge_index_down[:,range_down[l,0]:range_down[l,1]], edge_attr_down[range_down[l,0]:range_down[l,1],:])
                x = F.relu(x)

            #upward
            for l in reversed(range(self.level)):
                x = x + self.conv_list[l](x, edge_index_mid[:,range_mid[l,0]:range_mid[l,1]], edge_attr_mid[range_mid[l,0]:range_mid[l,1],:])
                x = F.relu(x)
                if l > 0:
                    x = x + self.conv_up_list[l-1](x, edge_index_up[:,range_up[l-1,0]:range_up[l-1,1]], edge_attr_up[range_up[l-1,0]:range_up[l-1,1],:])
                    x = F.relu(x)


        x = F.relu(self.fc_out1(x[:self.points[0]]))
        x = self.fc_out2(x)
        return x



TRAIN_PATH = 'data/piececonst_r241_N1024_smooth1.mat'
TEST_PATH = 'data/piececonst_r241_N1024_smooth2.mat'


for r in [8,6,4,2,1]:

    s = int(((241 - 1) / r) + 1)
    n = s ** 2
    k = 1

    print('!!!!!!!!!!!!!! s ', s, ' !!!!!!!!!!!!!!!!!!!!!!!!')

    m = [400, 100, 25]
    radius_inner = [0.5 / 4, 0.5 / 2, 0.5]
    radius_inter = [0.5 / 4 * 1.41, 0.5 / 2 * 1.41]


    level = len(m)
    print('resolution', s)

    ntrain = 100
    ntest = 100

    # don't change this
    batch_size = 1
    batch_size2 = 1

    width = 64
    ker_width = 256
    depth = 4
    edge_features = 6
    node_features = 6

    epochs = 200
    learning_rate = 0.1 / ntrain
    scheduler_step = 10
    scheduler_gamma = 0.8



    path = 'neurips3_multigraph_s'+str(s)+'_ntrain'+str(ntrain)+'_kerwidth'+str(ker_width) + 'r' + str(r)
    path_model = 'model/' + path
    path_train_err = 'results/' + path + 'train.txt'
    path_test_err61 = 'results/'+path+'test61.txt'
    path_test_err121 = 'results/'+path+'test121.txt'
    path_test_err241 = 'results/'+path+'test241.txt'
    path_runtime = 'results/' + path + 'time.txt'
    path_image = 'results/' + path

    runtime = np.zeros(2,)

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

    meshgenerator = RandomMultiMeshGenerator([[0, 1], [0, 1]], [s, s], level=level, sample_sizes=m)
    data_train = []
    for j in range(ntrain):
        for i in range(k):
            idx, idx_all = meshgenerator.sample()
            grid, grid_all = meshgenerator.get_grid()
            edge_index, edge_index_down, edge_index_up = meshgenerator.ball_connectivity(radius_inner, radius_inter)
            edge_index_range, edge_index_down_range, edge_index_up_range = meshgenerator.get_edge_index_range()
            edge_attr, edge_attr_down, edge_attr_up = meshgenerator.attributes(theta=train_a[j, :])
            x = torch.cat([grid_all, train_a[j, idx_all].reshape(-1, 1),
                           train_a_smooth[j, idx_all].reshape(-1, 1),
                           train_a_gradx[j, idx_all].reshape(-1, 1),
                           train_a_grady[j, idx_all].reshape(-1, 1)
                           ], dim=1)
            data_train.append(Data(x=x, y=train_u[j, idx[0]],
                                   edge_index_mid=edge_index, edge_index_down=edge_index_down,
                                   edge_index_up=edge_index_up,
                                   edge_index_range=edge_index_range, edge_index_down_range=edge_index_down_range,
                                   edge_index_up_range=edge_index_up_range,
                                   edge_attr_mid=edge_attr, edge_attr_down=edge_attr_down, edge_attr_up=edge_attr_up,
                                   sample_idx=idx[0]))

    print(x.shape)
    print(edge_index_range)
    print(edge_index_down_range)
    print(edge_index_up_range)

    print(edge_index.shape, edge_attr.shape)
    print(edge_index_down.shape, edge_attr_down.shape)
    print(edge_index_up.shape, edge_attr_up.shape)

    meshgenerator = RandomMultiMeshGenerator([[0, 1], [0, 1]], [241, 241], level=level, sample_sizes=m)
    data_test241 = []
    for j in range(ntest):
        for i in range(k):
            idx, idx_all = meshgenerator.sample()
            grid, grid_all = meshgenerator.get_grid()
            edge_index, edge_index_down, edge_index_up = meshgenerator.ball_connectivity(radius_inner, radius_inter)
            edge_index_range, edge_index_down_range, edge_index_up_range = meshgenerator.get_edge_index_range()
            edge_attr, edge_attr_down, edge_attr_up = meshgenerator.attributes(theta=test_a241[j, :])
            x = torch.cat([grid_all, test_a241[j, idx_all].reshape(-1, 1),
                           test_a_smooth241[j, idx_all].reshape(-1, 1),
                           test_a_gradx241[j, idx_all].reshape(-1, 1),
                           test_a_grady241[j, idx_all].reshape(-1, 1)
                           ], dim=1)
            data_test241.append(Data(x=x, y=test_u241[j, idx[0]],
                                  edge_index_mid=edge_index, edge_index_down=edge_index_down,
                                  edge_index_up=edge_index_up,
                                  edge_index_range=edge_index_range, edge_index_down_range=edge_index_down_range,
                                  edge_index_up_range=edge_index_up_range,
                                  edge_attr_mid=edge_attr, edge_attr_down=edge_attr_down, edge_attr_up=edge_attr_up,
                                  sample_idx=idx[0]))

    meshgenerator = RandomMultiMeshGenerator([[0, 1], [0, 1]], [121, 121], level=level, sample_sizes=m)
    data_test121 = []
    for j in range(ntest):
        for i in range(k):
            idx, idx_all = meshgenerator.sample()
            grid, grid_all = meshgenerator.get_grid()
            edge_index, edge_index_down, edge_index_up = meshgenerator.ball_connectivity(radius_inner, radius_inter)
            edge_index_range, edge_index_down_range, edge_index_up_range = meshgenerator.get_edge_index_range()
            edge_attr, edge_attr_down, edge_attr_up = meshgenerator.attributes(theta=test_a121[j, :])
            x = torch.cat([grid_all, test_a121[j, idx_all].reshape(-1, 1),
                           test_a_smooth121[j, idx_all].reshape(-1, 1),
                           test_a_gradx121[j, idx_all].reshape(-1, 1),
                           test_a_grady121[j, idx_all].reshape(-1, 1)
                           ], dim=1)
            data_test121.append(Data(x=x, y=test_u121[j, idx[0]],
                                  edge_index_mid=edge_index, edge_index_down=edge_index_down,
                                  edge_index_up=edge_index_up,
                                  edge_index_range=edge_index_range, edge_index_down_range=edge_index_down_range,
                                  edge_index_up_range=edge_index_up_range,
                                  edge_attr_mid=edge_attr, edge_attr_down=edge_attr_down, edge_attr_up=edge_attr_up,
                                  sample_idx=idx[0]))

    meshgenerator = RandomMultiMeshGenerator([[0, 1], [0, 1]], [61, 61], level=level, sample_sizes=m)
    data_test61 = []
    for j in range(ntest):
        for i in range(k):
            idx, idx_all = meshgenerator.sample()
            grid, grid_all = meshgenerator.get_grid()
            edge_index, edge_index_down, edge_index_up = meshgenerator.ball_connectivity(radius_inner,
                                                                                         radius_inter)
            edge_index_range, edge_index_down_range, edge_index_up_range = meshgenerator.get_edge_index_range()
            edge_attr, edge_attr_down, edge_attr_up = meshgenerator.attributes(theta=test_a61[j, :])
            x = torch.cat([grid_all, test_a61[j, idx_all].reshape(-1, 1),
                           test_a_smooth61[j, idx_all].reshape(-1, 1),
                           test_a_gradx61[j, idx_all].reshape(-1, 1),
                           test_a_grady61[j, idx_all].reshape(-1, 1)
                           ], dim=1)
            data_test61.append(Data(x=x, y=test_u61[j, idx[0]],
                                     edge_index_mid=edge_index, edge_index_down=edge_index_down,
                                     edge_index_up=edge_index_up,
                                     edge_index_range=edge_index_range,
                                     edge_index_down_range=edge_index_down_range,
                                     edge_index_up_range=edge_index_up_range,
                                     edge_attr_mid=edge_attr, edge_attr_down=edge_attr_down,
                                     edge_attr_up=edge_attr_up,
                                     sample_idx=idx[0]))

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader241 = DataLoader(data_test241, batch_size=batch_size2, shuffle=False)
    test_loader121 = DataLoader(data_test121, batch_size=batch_size2, shuffle=False)
    test_loader61 = DataLoader(data_test61, batch_size=batch_size2, shuffle=False)

    t2 = default_timer()

    print('preprocessing finished, time used:', t2-t1)
    device = torch.device('cuda')

    # print('use pre-train model')
    # model = torch.load('model/multigraph2241_n100')

    model = KernelInduced(width=width, ker_width=ker_width, depth=depth, ker_in=edge_features,
                          points=m, level=level, in_width=node_features,  out_width=1).cuda()

    # model = KernelInduced_SUM(width=width, ker_width=ker_width, depth=depth, ker_in=edge_features,
    #                       points=m, level=level, in_width=node_features,  out_width=1).cuda()


    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

    myloss = LpLoss(size_average=False)
    u_normalizer.cuda()
    ttrain = np.zeros((epochs, ))
    ttest241 = np.zeros((epochs,))
    ttest121 = np.zeros((epochs,))
    ttest61 = np.zeros((epochs,))
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
            # mse.backward()

            l2 = myloss(
                u_normalizer.decode(out.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)),
                u_normalizer.decode(batch.y.view(batch_size, -1), sample_idx=batch.sample_idx.view(batch_size, -1)))
            l2.backward()

            optimizer.step()
            train_mse += mse.item()
            train_l2 += l2.item()

        scheduler.step()
        t2 = default_timer()
        ttrain[ep] = train_l2 / (ntrain * k)

        print(ep, t2 - t1, train_mse / len(train_loader), train_l2 / (ntrain * k))

    runtime[0] =  t2 - t1


    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader241:
            batch = batch.to(device)
            out = model(batch)
            out = u_normalizer.decode(out.view(batch_size2, -1), sample_idx=batch.sample_idx.view(batch_size2, -1))
            test_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()
            # test_l2 += myloss(out.view(batch_size2,-1), y_normalizer.encode(batch.y.view(batch_size2, -1))).item()
    ttest241[ep] = test_l2 / ntest
    print(ep, '241', t2 - t1, test_l2 / ntest)
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader121:
            batch = batch.to(device)
            out = model(batch)
            out = u_normalizer.decode(out.view(batch_size2, -1), sample_idx=batch.sample_idx.view(batch_size2, -1))
            test_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()
            # test_l2 += myloss(out.view(batch_size2,-1), y_normalizer.encode(batch.y.view(batch_size2, -1))).item()
    ttest121[ep] = test_l2 / ntest
    print(ep, '121', t2 - t1, test_l2 / ntest)
    test_l2 = 0.0
    with torch.no_grad():
        for batch in test_loader61:
            batch = batch.to(device)
            out = model(batch)
            out = u_normalizer.decode(out.view(batch_size2, -1), sample_idx=batch.sample_idx.view(batch_size2, -1))
            test_l2 += myloss(out, batch.y.view(batch_size2, -1)).item()
            # test_l2 += myloss(out.view(batch_size2,-1), y_normalizer.encode(batch.y.view(batch_size2, -1))).item()
    ttest61[ep] = test_l2 / ntest
    print(ep, '61', t2 - t1, test_l2 / ntest)


    np.savetxt(path_train_err, ttrain)
    np.savetxt(path_test_err61, ttest61)
    np.savetxt(path_test_err121, ttest121)
    np.savetxt(path_test_err241, ttest241)
    torch.save(model, path_model)

