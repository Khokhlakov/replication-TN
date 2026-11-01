import numpy as np
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import pickle
import scipy
import sys

X = np.load("./X.npy")
Y = np.load("./Y.npy")
Y = Y.astype(float)


#label = np.load("./crosslabel.npy")
#label = np.asarray(label,dtype = int)
idx = np.load("./importanceidx.npy")

n_topfeature = 300

X = X[:,idx[0:n_topfeature]]


n_splits = 10
kf = KFold(n_splits=10,shuffle=True)

RMSD_whole = []
pearsonr_whole = []
iter_num = int(sys.argv[1])

n_num = X.shape[0]

for j in range(0,iter_num):

    result = np.zeros(n_num)
    print('iter_num',j)
    i = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
	
        i = i+1
        print('start training on Fold ',i)
        model = GradientBoostingRegressor(n_estimators=20000, max_features = "sqrt", learning_rate=0.001, max_depth=6, min_samples_split=3, subsample=0.4). fit(X_train, Y_train)

        result[test_index] = model.predict(X_test)
        RMSD = np.sqrt(mean_squared_error(Y_test,model.predict(X_test)))
        pearsonr = scipy.stats.pearsonr(Y_test,model.predict(X_test))
        print(RMSD)
        print(pearsonr)

    RMSD = np.sqrt(mean_squared_error(Y,result))
    pearsonr = scipy.stats.pearsonr(Y,result)


    print('RMSD=',RMSD)
    print('pearsonr=',pearsonr)

    RMSD_whole.append(RMSD)
    pearsonr_whole.append(pearsonr[0])

np.save("result",result)
np.save("rmse_whole",RMSD_whole)
np.save("pearsonr_whole",pearsonr_whole)

print("rmse median:", np.median(RMSD_whole))
print("pearsonr median:",np.median(pearsonr_whole))

file_out = open("crossvalidation_abbind.txt","wb")
file_out.write("Pearsonr_average is:")
file_out.write(str(np.mean(pearsonr_whole)))
file_out.write("\n")
file_out.write("RMSE_average is:")
file_out.write(str(np.mean(RMSD_whole)))
file_out.close()


