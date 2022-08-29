from MELMtrain import *
from MELMpredict import *

def MELM(trainX,trainY,testX,testY,option):

    # Train ELM
    [Model,TrainAcc,TrainingTime,Training_loss] = MELMtrain(trainX,trainY,option)


    # Using trained model, predict the testing data
    [TestAcc,TestingTime,Testing_loss] = MELMpredict(testX,testY,Model)

    return Model,TrainAcc,TestAcc,TrainingTime,TestingTime,Training_loss,Testing_loss
#EOF

