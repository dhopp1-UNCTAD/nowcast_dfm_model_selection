import numpy as np
import pandas as pd
from pmdarima.arima import auto_arima
from dateutil.relativedelta import relativedelta

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

def gen_arma(x, n_periods):
	"x: pd dataframe with date and value columns, n_periods is months ahead, including for quarterly variables, in between months registered as 0"
	x_est = x.loc[x.iloc[:,1] != 0.0,:].set_index("date")
	arma = auto_arima(x_est, start_p=0, d=0, start_q=0, D=0, stationary=True)
	preds = list(arma.predict(n_periods=n_periods))
	to_be_filled = [x.loc[len(x)-1,"date"] + relativedelta(months=i+1) for i in range(n_periods)]
	# quarterly series
	if set(x.loc[x.iloc[:,1] != 0.0,"date"].dt.month) <= set([3,6,9,12]):
		actually_filled = [max(x_est.index) + relativedelta(months=i*3+3) for i in range(n_periods)] # months actually filled in with n_periods
		actually_filled_df = pd.DataFrame({"date":actually_filled, x_est.columns[0]:preds})
		preds = pd.DataFrame({"date":to_be_filled}).merge(actually_filled_df, how="left", on="date").fillna(0.0) # merging with what actually should be filled in on a monthly basis
		preds = list(preds.iloc[:,1])
	return preds

def gen_vintage_dataset(rawdata, sim_month):
	"generate a simulated vintage/ragged dataset based on actual publication lags"
	
	catalog = pd.read_csv("/home/danhopp/dhopp1/UNCTAD/nowcast_data_update/helper/catalog.csv")
	dataset = rawdata.loc[rawdata.date <= sim_month,:]
	
	for col in dataset.columns[1:]:
		lag = catalog.loc[catalog.code == col, "publication_lag"].values[0]
		dataset.loc[len(dataset)-lag-1:, col] = 0.0 # this will have to be replaced with function to do ARMA if necessary
	
	return dataset
	
def gen_ragged_dataset(dataset, variables, month_lag, missings, rawdata):
	"adjust an already created LSTM dataset back some months, missings=value to fill with missing, or 'ARMA' to fill with auto arima"
	
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
				
			# for ARMA estimation
			if missings == "ARMA":
				full_series = rawdata.loc[:, ["date", variables[var]]].fillna(0.0) # full original series of this variable
				non_missing = [i for i, e in enumerate(X_ragged[obs, :, var]) if e != 0.0][-1] # last period non-missing
				n_periods = len(X_ragged[obs, :, var]) - non_missing # # of missing periods
				subseries = list(X_ragged[obs, :, var][:non_missing+1]) # cutting off trailing missings
				placement = [x for x in range(len(full_series.iloc[:,1])) if list(full_series.iloc[:,1])[x:x+len(subseries)] == subseries][0] + len(subseries) # getting the original series up until the missing data point
				est_series = full_series.iloc[:placement,:] # the cut down series to estimate ARMA on
				filling = gen_arma(est_series, n_periods) # estimating the values to fill missings with
				X_ragged[obs, -n_periods:, var] = filling

	return X_ragged