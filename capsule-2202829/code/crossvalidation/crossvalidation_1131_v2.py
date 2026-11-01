import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pickle
import scipy
import sys

X1 = np.load("X_skempi_feature1.npy")
X2 = np.load("X_skempi_feature2.npy")

X = np.concatenate((X1,X2),axis=1)
Y = np.load("Y_skempi.npy")
Y = Y.astype(float)


print(Y.shape)
n_num = Y.shape[0]


print X.shape

n_split = 10

RMSE_whole = []
pearsonr_whole = []
iter_num = int(sys.argv[1])

for j in range(0,iter_num):

    result = np.zeros(n_num)
    kf = KFold(n_splits=10,shuffle=True)
    
    pear_sum = 0

    for train_index, test_index in kf.split(X):
        print test_index

        X_train,X_test = X[train_index], X[test_index]
        Y_train,Y_test = Y[train_index], Y[test_index]
        print('start training')
        model = GradientBoostingRegressor(n_estimators=20000, max_features = "sqrt", learning_rate=0.001, max_depth=6, min_samples_split=3, subsample=0.4). fit(X_train, Y_train)

        result[test_index] = model.predict(X_test)
        RMSD = np.sqrt(mean_squared_error(Y_test,model.predict(X_test)))
        pearsonr = scipy.stats.pearsonr(Y_test,model.predict(X_test))

        pear_sum = pear_sum + pearsonr[0]
        print RMSD
        print pearsonr

    RMSD = np.sqrt(mean_squared_error(Y,result))
    pearsonr = scipy.stats.pearsonr(Y,result)

    
    pearsonr_whole.append(pearsonr[0])
    RMSE_whole.append(RMSD)

np.save("pearsonr_whole",pearsonr_whole)




file_out = open("crossvalidation_skempi.txt","wb")
file_out.write("Pearsonr_average is:")
file_out.write(str(np.mean(pearsonr_whole)))
file_out.write("\n")
file_out.write("RMSE_average is:")
file_out.write(str(np.mean(RMSE_whole)))
file_out.close()
