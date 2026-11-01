import sys,os
import numpy as np
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D
from keras.layers.merge import Concatenate


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pickle
import scipy

import h5py
from keras.models import load_model

def normalize(X):
    mean = np.mean(X,axis=0)
    std = np.std(X,axis=0)
    length1 = X.shape[0]
    X_train_normed = X

    for i in range(0,length1):
        for j in range(0,48):
            for k in range(0,108):
                if std[j,k]!=0 :
                    X_train_normed[i,j,k] = (X_train_normed[i,j,k]-mean[j,k])/std[j,k]
    return X_train_normed


def main():
    option =sys.argv[1]

    model_name = './model/cnn_model.h5'
    model = load_model(model_name)
    model_cnn_feature = Model(inputs = model.input, outputs = model.layers[7].output)


    X_train = np.load("./model/X_unnormed.npy")
    X = np.load("Feature_cnn.npy")
    X = np.reshape(X,(1,48,108))
    X_train = np.concatenate((X,X_train),axis=0)
    X = normalize(X_train)

    cnn_feature = model_cnn_feature.predict(X)[0,:]
    
    idx = np.load("./model/importance_idx.npy")
    cnn_feature = cnn_feature[idx[:300]]


    if option == '0':
        print "Using topo and aux feautures"
        gbdt_model = pickle.load(open("../../data/gbdt1.pickle","rb"))
        X_12  = np.load("X_h12.npy")
        X_aux = np.load("X_aux.npy")
        X = np.concatenate((cnn_feature,X_12,X_aux),axis=0)
        X = np.reshape(X,(1,-1))
        ddg = gbdt_model.predict(X)
        
    else:
        print "Using only topo features"
        gbdt_model = pickle.load(open("../../data/gbdt2.pickle","rb"))
        X_12 = np.load("X_h12.npy")
        X = np.concatenate((cnn_feature,X_12),axis=0)
        X = np.reshape(X,(1,-1))
        ddg = gbdt_model.predict(X)

    file_out = open("result.ddg","w")
    file_out.write("Prediction of binding affinity change is ")
    file_out.write(str(ddg[0]))
    file_out.write(" kcal/mol")
        
main()
