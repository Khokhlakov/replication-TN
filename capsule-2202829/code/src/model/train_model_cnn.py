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


def load_data():
	datafolder = './'
	X = np.load(datafolder + "X_cnn.npy")
	X = np.asarray(X,dtype = float)
	Y = np.load(datafolder + "Y_cnn.npy")
	Y = np.asarray(Y,dtype = float)
	return X,Y

def deep_model(X,Y,epoch_size):

	input1 = Input(shape =((48,108)))

    # Set up 1dconv  model
	model_con1d = Conv1D(16,3,padding='same',activation='relu',kernel_initializer='he_normal')(input1)
	model_con1d = Conv1D(16,3,padding='same',activation='relu',kernel_initializer='lecun_uniform')(model_con1d)
	model_con1d = Dropout(0.3)(model_con1d)
	model_con1d = Conv1D(32,3,padding='same',activation='relu',kernel_initializer='lecun_uniform')(model_con1d)
	model_con1d = Conv1D(32,3,padding='same',activation='relu',kernel_initializer='lecun_uniform')(model_con1d)
	model_con1d = Dropout(0.3)(model_con1d)
	intermediate_output = Flatten()(model_con1d)
	model_final = Dropout(0.1)(intermediate_output)
 	final_output = Dense(1,activation='linear')(model_final)	
	
	# Set up model input and output
	Model_deep = Model(inputs = input1, outputs = final_output)
	adam = Adam(lr=0.0001)
	Model_deep.compile(optimizer=adam,loss='mse',metrics = ['mean_squared_error'])
	Model_deep.fit(X, Y , batch_size = 8, epochs = epoch_size, shuffle = True)

	# Set up cnn_feature output model
	Model_cnn_feature = Model(inputs = Model_deep.input, outputs = Model_deep.layers[7].output)	

	return Model_deep,Model_cnn_feature

def main():

    X,Y = load_data()
    n_num = Y.shape[0]

    model,model_cnn_feature = deep_model(X,Y,2000)
    model.save("cnn_model.h5")
    model_cnn_feature.save("cnn_feature_model.h5")
    print "model saved" 


main()	
