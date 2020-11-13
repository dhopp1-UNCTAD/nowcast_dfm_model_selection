import numpy as np
import pandas as pd

def gen_raw_data(target_variable, variables=[]):
	"reading in raw data from nowcast data"
	
	rawdata = pd.read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/output/2020-11-10_database_tf.csv", parse_dates=["date"])
	catalog = pd.read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/helper/catalog.csv")
	if variables == []:
		variables = list(catalog.loc[catalog[target_variable] > 0,"code"])
	variables = ["date"] + [target_variable] + variables
	variables = pd.unique(variables)
	data = rawdata.loc[:,variables]
	data = data.fillna(0.0) # will have to be replaced with filling this with ARMA
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
	X = X[y != 0.0,:,:] # delete na ys
	y = y[y != 0.0]
	return X, y

def gen_vintage_dataset(rawdata, sim_month):
	"generate a simulated vintage/ragged dataset based on actual publication lags"
	
	catalog = pd.read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/helper/catalog.csv")
	dataset = rawdata.loc[rawdata.date <= sim_month,:]
	
	for col in dataset.columns[1:]:
		lag = catalog.loc[catalog.code == col, "publication_lag"].values[0]
		dataset.loc[len(dataset)-lag-1:, col] = 0.0 # this will have to be replaced with function to do ARMA if necessary
	
	return dataset
	
def gen_ragged_dataset(dataset, variables, month_lag):
	"adjust an already created LSTM dataset back some months"
	
	catalog = pd.read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/helper/catalog.csv")
	
	# get publication lags
	pub_lags = []
	for var in variables:
		lag = catalog.loc[catalog.code == var, "publication_lag"].values[0]
		pub_lags.append(lag)
		
	# generate ragged dataset
	X_ragged = np.array(dataset)		
	for obs in range(X_ragged.shape[0]): # go through every observation
		for var in range(len(pub_lags)): # every variable (and its corresponding lag)
			for ragged in range(1, pub_lags[var]+1-month_lag): # setting correct lags (-month_lag because input -2 for -2 months, so +2 additional months of lag)
				X_ragged[obs, X_ragged.shape[1]-ragged, var] = 0.0 # setting to missing data

	return X_ragged