
import h5py
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pickle




# train1 = np.loadtxt("results/RNN_nik_5layer_r16_w64_Admm_train.txt")
# train2 = np.loadtxt("results/MP_nik_5layer_r16_w64_Admm_train.txt")
# test1 = np.loadtxt("results/RNN_nik_5layer_r16_w64_Admm_test.txt")
# test2 = np.loadtxt("results/MP_nik_5layer_r16_w64_Admm_test.txt")


# train1 = pickle.load(open("results/RNN_nik_4layer_r16_w64_Admm_train.txt", "rb"))
# train2 = pickle.load(open("results/MPdiag_nik_4layer_r16_w64_Admm_train.txt", "rb"))
# train3 = pickle.load(open("results/MPdense_nik_4layer_r16_w64_Admm_train.txt", "rb"))
# train4 = pickle.load(open("results/MPgauss_nik_4layer_r16_w64_Admm_train.txt", "rb"))
n1 = 25
n2 = 400

train1 = np.loadtxt("results/RNN3_r16_data10_train.txt")[n1:n2]
train2 = np.loadtxt("results/RNN3_r16_data100_train.txt")[n1:n2]
train3 = np.loadtxt("results/RNN3_r16_data1000_train.txt")[n1:n2]

# test1 = np.loadtxt("results/MPaug_r16_data10full_test.txt")[n1:n2]
# test2 = np.loadtxt("results/MPaug_r32_data10full_test.txt")[n1:n2]
# test3 = np.loadtxt("results/MPaug_r64_data10full_test.txt")[n1:n2]

# test1 = np.loadtxt("results/MPdense_nik_4layer_r16_w64_Admm_train.txt")
# test2 = np.loadtxt("results/MPdense_nik_4layer_r16_w64_Admm_train.txt")

# train1 = np.log(train1)[10:]
# train2 = np.log(train2)[10:]
# train3 = np.log(train3)[10:]
# train4 = np.log(train4)[10:]
# test1 = np.log(test1)
# test2 = np.log(test2)

# np.savetxt("results/RNN_nik_4layer_r16_w64_Admm_train.txt", train1)
# np.savetxt("results/MPdiag_nik_4layer_r16_w64_Admm_train.txt", train2)
# np.savetxt("results/MPdense_nik_4layer_r16_w64_Admm_train.txt", train3)
# np.savetxt("results/MPgauss_nik_4layer_r16_w64_Admm_train.txt", train4)



# test = np.loadtxt(path + "/test_loss_net.txt")
# test_u = np.loadtxt(path + "/test_loss_u_net.txt")
# test_mp = np.loadtxt(path + "/test_loss_mp1_net.txt")
# test_su = np.loadtxt(path + "/test_loss_su_net.txt")
#test_fs = np.loadtxt(path + "/test_loss_fs_net.txt")

plt.plot(train1, label='10 equations')
plt.plot(train2, label='100 equations')
plt.plot(train3, label='1000 equations')
# plt.plot(test1, label='r16 test')
# plt.plot(test2, label='r32 test')
# plt.plot(test3, label='r64 test')

# plt.plot(test2, label='test Edge GNN')
# # plt.plot(test_u, label='fc + u-net')
# # plt.plot(test_mp, label='edge')
# # plt.plot(test_su, label='sep + u-net')
# #plt.plot(test_fs, label='fc + u-net')
plt.legend(loc='upper right')
plt.show()
