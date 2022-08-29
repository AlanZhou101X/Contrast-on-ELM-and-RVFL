# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/5/8 1:36
@author: merci
"""
import os
import numpy as np
from option import option as op
from MELM import *
import time
import scipy.io as sio
from Logger import Logger
import sys
np.seterr(divide='ignore', invalid='ignore')

root_path = '../../dataset'
# data_name = 'led-display'
# n_device = 0
# print('Dataset Name:{}\nDevice Number:#CPU'.format(data_name))
sys.stdout = Logger(file_path="/result")
# cp.cuda.Device(n_device).use()

data1 = sio.loadmat(root_path + '/train_5403.mat')
# data1 = sio.loadmat(root_path+'/DATA2.mat')
data2 = data1['sigXY']

train_acc_list = []
pred_acc_list = []
train_loss_list = []
pred_loss_list = []
train_time_list = []


def get_avg(list):
    sum = 0
    for l in range(0, len(list)):
        sum = sum + list[l]
    return sum / len(list)


for i in range(0, 20):
    print("Running NO.:" + str(i + 1))
    data = data2[:, :16]
    label = data2[:, 16:]

    shuffle_index = np.arange(len(data2))
    np.random.shuffle(shuffle_index)
    train_number = int(0.7 * len(data2))
    train_index = shuffle_index[:train_number]
    val_index = shuffle_index[train_number:]

    train_X = data[train_index]
    train_Y = label[train_index]
    test_X = data[val_index]
    test_Y = label[val_index]

    n_types = len(train_Y[0])
    n_CV = 0

    # One hot coding for the target
    # dataY_tmp = cp.zeros((trainY.size, n_types))
    # for i in range(n_types):
    #     for j in range(trainY.size):  # remove this loop
    #         if trainY[j] == types[i]:
    #             dataY_tmp[j, i] = 1

    option = op(N=2000, L=8, C=2048, scale=1, seed=0, nCV=0, n_types=n_types)
    # N_range = [256, 512, 1024]
    # N_range = [64, 128, 256, 512]
    # N_range = [16, 32, 64]
    N_range = [2000]
    # L = 32
    L = 8
    option.scale = 1
    # C_range = np.append(0,2.**np.arange(-6, 12, 2))
    # C_range = 2.**np.arange(-6, 12, 2)
    C_range = [2048]

    Models_tmp = []
    Models = []

    train_acc_result = np.zeros((n_CV, 1))
    test_acc_result = np.zeros((n_CV, 1))
    train_time_result = np.zeros((n_CV, 1))
    test_time_result = np.zeros((n_CV, 1))

    MAX_acc = 0
    option_best = op(N=2000, L=8, C=2048, scale=1, seed=0, nCV=0, n_types=n_types)

    st = time.time()
    for n in N_range:
        option.N = n
        for j in C_range:
            option.C = j
            for l in range(1, L):
                sto = time.time()
                option.L = l
                [model_tmp, train_acc_temp, test_acc_temp, training_time_temp, testing_time_temp, training_loss_temp,
                 testing_loss_temp] = MELM(train_X, train_Y, test_X, test_Y, option)
                if test_acc_temp > MAX_acc:
                    MAX_acc = test_acc_temp
                    option_best.acc_test = test_acc_temp.max()
                    option_best.acc_train = train_acc_temp.max()
                    option_best.C = option.C
                    option_best.N = option.N
                    option_best.L = option.L
                    # option_best.nCV = i
                    print('Temp Best Option:{}'.format(option_best.__dict__))
                print('Training Time for one option set:{:.2f}'.format(time.time() - sto))
                if time.time() - sto > 10:
                    print('current settings:{}'.format(option.__dict__))
    [model_MELM, train_acc0, test_acc0, train_time0, test_time0, training_loss0, testing_loss0] = MELM(train_X, train_Y,
                                                                                                       test_X, test_Y,
                                                                                                       option_best)
    print('Training Time for one fold set:{:.2f}'.format(time.time() - st))
    Models.append(model_MELM)

    train_acc_list.append(train_acc0)
    pred_acc_list.append(test_acc0)
    train_time_list.append(train_time0)
    train_loss_list.append(training_loss0)
    pred_loss_list.append(testing_loss0)
    del model_MELM
    print('Best Train accuracy in:{}\nBest Test accuracy in:{}'.format(train_acc_result,
                                                                       test_acc_result))
    mean_train_acc = train_acc_result.mean()
    mean_test_acc = test_acc_result.mean()
    print('Train accuracy:{}\nTest accuracy:{}'.format(train_acc_result, test_acc_result))
    print('Mean train accuracy:{}\nMean test accuracy:{}'.format(mean_train_acc, mean_test_acc))
    print("-----------------------------------------------------------------------------------")

print("-----------------------------------------------------------------------------------")
print("train_acc_avg:", get_avg(train_acc_list))
print("pred_acc_avg:", get_avg(pred_acc_list))
print("train_loss_avg:", get_avg(train_loss_list))
print("pred_loss_avg:", get_avg(pred_loss_list))
print("train_time_avg:", get_avg(train_time_list))
