import numpy as np
import time
from function import *
from majorityVoting import *
from sklearn.metrics import log_loss


def one_hot(x, n_class):
    # print(n_class)
    y = np.zeros([len(x), n_class])
    for i in range(len(x)):
        y[i, x[i]] = 1
    return y


def MELMpredict(testX,testY,model):

    [Nsample,Nfea]= testX.shape
    w=model.w
    b=model.b
    beta=model.beta
    mu=model.mu
    sigma=model.sigma
    L= model.L
    ada_weights = model.ada_weights
    n_types=model.n_types
    A=[]
    A_input=testX

    time_start=time.time()

    for i in range(L):
        A1=np.matmul(A_input,w[i])
        A1 = (A1-mu[i])/sigma[i]
        A1 = A1 + np.repeat(b[i], Nsample, 0)
        A1=relu(A1)
        # A1=selu(A1)
        #A1_temp1 = np.concatenate([testX,A1,np.ones((Nsample,1))],axis=1)
        A1_temp1 = np.concatenate([A1, np.ones((Nsample, 1))], axis=1)


        A.append(A1_temp1)

        #clear A1 A1_temp1 A1_temp2 beta1
        #A_input = np.concatenate([testX,A1],axis=1)
        A_input = A1

    samm_prob =[]
    for i in range(L):
        A_temp=A[i]
        beta_temp=beta[i]
        testY_temp=np.matmul(A_temp,beta_temp)
        n_classes = np.unique(testY).size
        prob = np.expand_dims(softmax(testY_temp), axis=1)
        h = (n_classes - 1) * (np.log(prob) - (1. / n_classes) * np.log(prob).sum(axis=1)[:, np.newaxis])
        samm_prob.append(h)
    pred = sum(samm_prob) / L
    #print("Nsample",Nsample)
    # print("pred", np.argmax(pred, axis=2).ravel())
    # print("label", np.argmax(testY, axis=1).ravel())
    pred = np.argmax(pred, axis=2).ravel()
    pred1 = one_hot(pred, n_types)
    TestingAccuracy = np.sum(pred == np.argmax(testY, axis=1).ravel()) / Nsample

    time_end = time.time()

    Testing_time = time_end - time_start

    Testing_loss = log_loss(testY, pred1)

    return TestingAccuracy,Testing_time,Testing_loss


