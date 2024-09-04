import numpy as np
import pandas as pd
import os
from numpy import matlib
from numpy.random import uniform
from AZO import AZO
from EOO import EOO
from GSO import GSO
from Model_CR_LSTM_AM import Model_CR_LSTM_AM
from Model_DTCN import Model_DTCN
from Model_MOBILENET import Model_MOBILENET
from Model_RNN import Model_RNN
from Objective_Function import Objective_Function
from Global_vars import Global_vars
from Model_RBM import Model_RBM_Feat
from Model_Autoencoder import Model_Autoencoder_Feat
from Model_1DCNN import Model_1DCNN_Feat
from PROPOSED import PROPOSED
from RSA import RSA
from Plot_Results import Plot_Convergence, Plot_Results, Plot_Confusion, Plot_ROC

## READ DATASETS AND PRE-PROCESSING ##
# Read the dataset
an = 0
if an == 1:
    Datas = []
    Target = []
    dir = "./Datasets/"
    dirs = os.listdir(dir)
    for i in range(len(dirs)):  # loop to read all dataset
        file = dirs[i]
        if '.csv' in file:
            df = pd.read_csv(dir + file)
            np_array = np.asarray(df)
        if '.xlsx' in file:
            df = pd.read_excel(dir + file)
            np_array = np.asarray(df)
        elif '.txt' in file:
            File_data = np.loadtxt(dir + file)
            if i == 3:
                data = File_data[:, :-1]
                tar = File_data[:, -1]
            else:
                data = File_data[:, 1:]
                tar = File_data[:, 0]
        if i == 0:
            tar = np_array[:, np_array.shape[1] - 1]
            data = np_array[:, :np_array.shape[1] - 1]
        elif i == 2:
            tar = np_array[:, 1]
            df = df.drop('Label', axis=1)
            np_array = np.asarray(df)
            data = np_array
        elif i == 4:
            tar = np_array[:, -1]
            df = df.drop('label_mcc', axis=1)
            np_array = np.asarray(df)
            data = np_array
        tar = np.asarray(tar).astype('int')
        if len(np.unique(tar)) > 2:
            uniq = np.unique(tar)
            Tar = np.zeros((len(tar), len(uniq))).astype('int')
            for iter in range(len(uniq)):
                Tar[tar == uniq[iter], iter] = 1
        else:
            Tar = tar.reshape(-1, 1)
        Datas.append(data)
        Target.append(Tar)
    np.save('Datas.npy', Datas)
    np.save('Target.npy', Target)


# Data Cleaning
an = 0
if an == 1:
    Datas = []
    Targets = []
    Data = np.load('Datas.npy', allow_pickle=True)
    for i in range(len(Data)):                          # loop to read all dataset
        data = Data[i]
        pd.isnull('data')                               # locates missing data
        df = pd.DataFrame(data)
        # Replace with 0 values. Accepts regex.
        df.replace(np.NAN, 0, inplace=True)
        # Replace with zero values
        df.fillna(value=0, inplace=True)
        df.drop_duplicates()                            # removes the duplicates
        data = np.array(df)
        Datas.append(data)
    np.save('Data.npy', Datas)

## Deep Feature Extraction
an = 0
if an == 1:
    oneDCNN_Features=[]
    RBM_Features=[]
    Autoencoder_Features=[]
    Cleaned_Data = np.load('Data.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    for i in range(len(Target)):
        RBM_Feat = Model_RBM_Feat(Cleaned_Data[i], Target[i])
        Autoencoder_Feat = Model_Autoencoder_Feat(Cleaned_Data[i], Target[i])
        one1DCNN_Feat = Model_1DCNN_Feat(Cleaned_Data[i], Target[i])
        RBM_Features.append(RBM_Feat)
        Autoencoder_Features.append(Autoencoder_Feat)
        oneDCNN_Features.append(one1DCNN_Feat)
    RBM_Feat = np.array(RBM_Features, dtype='object')
    Autoencoder_Feat = np.array(Autoencoder_Features, dtype='object')
    oneDCNN_Feat = np.array(oneDCNN_Features, dtype='object')
    np.save('RBM_Features.npy', RBM_Feat)
    np.save('Autoencoder_Features.npy', Autoencoder_Feat)
    np.save('oneDCNN_Features.npy', oneDCNN_Feat)


## OPTIMIZATION FOR Deep weighted feature fusion
an = 0
if an == 1:
    BestSol=[]
    Fitness=[]
    RBM_Feat = np.load('RBM_Features.npy', allow_pickle=True)
    Autoencoder_Feat = np.load('Autoencoder_Features.npy', allow_pickle=True)
    oneDCNN_Feat = np.load('oneDCNN_Features.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    for i in range(len(Target)): # For all Datasets
        Global_vars.RBM_Feat = RBM_Feat[i]
        Global_vars.Autoencoder_Feat = Autoencoder_Feat[i]
        Global_vars.oneDCNN_Feat = oneDCNN_Feat[i]
        Global_vars.Target = Target[i]

        Npop = 10
        Chlen = RBM_Feat[i].shape[-1]+Autoencoder_Feat[i].shape[-1]+oneDCNN_Feat[i].shape[-1]
        xmin = matlib.repmat(np.concatenate([0.01*np.ones((1, Chlen))], axis=None), Npop,
                             1)
        xmax = matlib.repmat(np.concatenate([0.99*np.ones((1, Chlen))], axis=None), Npop, 1)
        initsol = np.zeros((xmax.shape))
        for p1 in range(Npop):
            for p2 in range(xmax.shape[1]):
                initsol[p1, p2] = uniform(xmin[p1, p2], xmax[p1, p2])
        fname = Objective_Function
        Max_iter = 50

        print("EOO...")
        [bestfit1, fitness1, bestsol1, time1] = EOO(initsol, fname, xmin, xmax, Max_iter)

        print("GSO...")
        [bestfit2, fitness2, bestsol2, time2] = GSO(initsol, fname, xmin, xmax, Max_iter)

        print("RSA...")
        [bestfit3, fitness3, bestsol3, time3] = RSA(initsol, fname, xmin, xmax, Max_iter)

        print("AZO...")
        [bestfit4, fitness4, bestsol4, time4] = AZO(initsol, fname, xmin, xmax, Max_iter)

        print("Enhanced AZO..")
        [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

        BestSol.append([bestsol1.ravel(), bestsol2.ravel(), bestsol3.ravel(), bestsol4.ravel(), bestsol5.ravel()])
        Fitness.append([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    BestSol = np.array(BestSol, dtype='object')
    Fitness = np.array(Fitness, dtype='object')
    np.save('Best_Sol.npy', BestSol)
    np.save('Fitness.npy', Fitness)


## PREDICTION OF FAULTS ##
an = 0
if an == 1:
    Eval_all = []
    Cleaned_Data = np.load('Data.npy', allow_pickle=True)
    RBM_Feat = np.load('RBM_Features.npy', allow_pickle=True)
    Autoencoder_Feat = np.load('Autoencoder_Features.npy', allow_pickle=True)
    oneDCNN_Feat = np.load('oneDCNN_Features.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    BestSol = np.load('Best_Sol.npy', allow_pickle=True)
    k_fold = 5
    for i in range(len(Target)):
        Ev = []
        for j in range(k_fold):
            Eval = np.zeros((10, 8), dtype=object)
            Total_Index = np.arange(Target[i].shape[0])
            Test_index = np.arange(((i - 1) * (Target[i].shape[0] / k_fold)) + 1, i * (Target[i].shape[0] / k_fold))
            Train_Index = np.setdiff1d(Total_Index, Test_index)

            for k in range(5):  # For all Algorithms
                Deep_Fused_Features = np.concatenate(
                    (RBM_Feat[i], Autoencoder_Feat[i], oneDCNN_Feat[i]), axis=-1)
                Weighted_Deep_Fused_Features = Deep_Fused_Features * BestSol[i][k]

                train_data = Weighted_Deep_Fused_Features[Train_Index, :]
                train_target = Target[i][Train_Index, :]
                test_data = Weighted_Deep_Fused_Features[Test_index, :]
                test_target = Target[i][Test_index, :]
                Eval[k, :] = Model_CR_LSTM_AM(train_data, train_target, test_data, test_target) # With Optimization

            train_data = Cleaned_Data[i][Train_Index, :]
            test_data = Cleaned_Data[i][Test_index, :]
            Eval[5, :] = Model_MOBILENET(train_data, train_target, test_data, test_target)
            Eval[6, :] = Model_DTCN(train_data, train_target, test_data, test_target)
            Eval[7, :] = Model_RNN(train_data, train_target, test_data, test_target)
            Eval[8, :] = Model_CR_LSTM_AM(train_data, train_target, test_data, test_target) # Without Optimization
            Eval[9, :] = Eval[4, :]
            Ev.append(Eval)
        Eval_all.append(Ev)
    np.save('Eval_all.npy', Eval_all)

Plot_Convergence()
Plot_Results()
Plot_Confusion()
Plot_ROC()

