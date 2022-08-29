# -*- coding: utf-8 -*-
# <nbformat>2</nbformat>

# <codecell>

# Demo python notebook for sklearn elm and random_hidden_layer modules
#
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD

# <codecell>
import numpy as np
import time

from elm import ELMClassifier
from Logger import Logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import scipy.io as sio
import sys
# <codecell>


def get_avg(list):
    sum = 0
    for l in range(0, len(list)):
        sum = sum + list[l]
    return sum / len(list)

# <codecell>

stdsc = StandardScaler()
sys.stdout = Logger(file_path="/result")
# file_path1 = '../../../dataset/train5403.mat'
file_path1 = '../../../dataset/DATA2.mat'
data1 = sio.loadmat(file_path1)
data2 = data1['sigXY']

train_acc_list = []
pred_acc_list = []
train_loss_list = []
pred_loss_list = []
train_time_list = []

for i in range(0, 20):
    print("Running NO.:"+str(i+1))
    data = data2[:, :16]
    label = data2[:, 16:]
    label = np.argmax(label, axis=1)
    dgx, dgy = stdsc.fit_transform(data), label
    dgx_train, dgx_test, dgy_train, dgy_test = train_test_split(dgx, dgy, test_size=0.3)
    elmc = ELMClassifier(n_hidden=500, activation_func='relu', alpha=0.9, random_state=0)
    #st = time.time()
    train_time = elmc.fit(dgx_train, dgy_train)
    #train_time = time.time() - st
    print('Training Time:{:.2f}'.format(train_time))
    train_time_list.append(train_time)
    train_accuracy,cross_entropy = elmc.score(dgx_train, dgy_train)
    print('train_accuracy:', train_accuracy)
    train_acc_list.append(train_accuracy)
    print('train_cross_entropy:', cross_entropy)
    train_loss_list.append(cross_entropy)

    pred_accuracy,pred_cross_entropy = elmc.score(dgx_test, dgy_test)
    print('pred_accuracy:', pred_accuracy)
    pred_acc_list.append(pred_accuracy)
    print('pred_cross_entropy:', pred_cross_entropy)
    pred_loss_list.append(pred_cross_entropy)
    print("------------------------------------------------------")

print("-----------------------------------------------------")
print("train_acc_avg:", get_avg(train_acc_list))
print("pred_acc_avg:", get_avg(pred_acc_list))
print("train_loss_avg:", get_avg(train_loss_list))
print("pred_loss_avg:", get_avg(pred_loss_list))
print("train_time_avg:", get_avg(train_time_list))