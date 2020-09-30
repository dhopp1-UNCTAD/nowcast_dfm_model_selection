import numpy as np
import pandas as pd

def gen_raw_data(target_variable, variables=[]):
	"reading in raw data from nowcast data"
	
	rawdata = pd.read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/2020-09-29_database_tf.csv", parse_dates=["date"])
	catalog = pd.read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/helper/catalog.csv")
	if variables == []:
		variables = list(catalog.loc[catalog[target_variable] > 0,"code"])
	variables = ["date"] + [target_variable] + variables
	variables = pd.unique(variables)
	data = rawdata.loc[:,variables]
	data = data.fillna(-1)
	return data

def gen_dataset(rawdata, target_variable, variables):
	"generate a raw dataset to feed into the model, outputs np array of dimensions `n_obs x n_features + 1`"
	
	data_dict = {}
	for variable in variables:
		data_dict[variable] = np.array(rawdata.loc[:,variable])
		data_dict[variable] = data_dict[variable].reshape((len(data_dict[variable]), 1))
	target = np.array(rawdata.loc[:,target_variable])
	target = target.reshape((len(target), 1))
	dataset = np.hstack(([data_dict[k] for k in data_dict] + [target]))
	return dataset

def split_sequences(sequences, n_steps):
	"""
	split a multivariate sequence into samples
	
	sequence: numpy array of dimensions `n_obs x n_features+1`, array of target values in last column
	n_steps: number of trailing datapoints (i.e. months) that go into each observation
	
	return: numpy tuple of:
		X: `n_obs x n_steps x n_features`
		y: `n_obs`
	"""
	X, y = list(), list()
	for i in range(len(sequences)):
        # find the end of this pattern
		end_ix = i + n_steps
        # check if we are beyond the dataset
		if end_ix > len(sequences):
			break
        # gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

def gen_x_y(dataset, n_timesteps):
	"generate the final dataset to feed the model"
	
	X, y = split_sequences(dataset, n_timesteps)
	X = X[y != -1,:,:] # delete na ys
	y = y[y != -1]
	return X, y
