import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
import helper.data_setup as data_setup
import helper.mv_net as mv_net
import helper.model_training as model_training

def run_model(variables, target_variable, n_timesteps=12, train_episodes=200, batch_size=30, lr=1e-2, decay=0.96, missings=0.0, n_hidden=20, n_layers=2, eval_df=pd.DataFrame()):
	"""
	train and evaluate a model
	
	variables: which variables to include in the model
	target_variable: e.g. "x_world"
	n_timesteps: number of trailing datapoints (i.e. months) that go into each observation
	train_episodes: # of epochs to train
	batch_size: # of samples per batch
	lr: initial learning rate
	decay: gamma for the learning rate decay
	missings: value to fill missings, "ARMA" for ARMA time series calculation of missings
	n_hidden: number of hidden states in the network architecture
	n_layers: number of stacked LSTM layers
	
	return: 
		convergence_plot: training data convergence plot
		eval_plot: plot of performance on test data at different lags
		eval_df: df with results of run
	"""
	
	### initial data set up
	tmp = data_setup.gen_raw_data(target_variable, variables)
	
	
	### setting up test and train data
	train = tmp.loc[tmp.date <= "2016-12-01",:]
	test = tmp.loc[(tmp.date >= "2016-01-01"),:]
	
	train_dataset = data_setup.gen_dataset(train, target_variable, variables)
	X, y = data_setup.gen_x_y(train_dataset, n_timesteps)
	
	# different test datasets for different month lags
	test_dataset = data_setup.gen_dataset(test, target_variable, variables)
	X_test, y_test = data_setup.gen_x_y(test_dataset, n_timesteps)
	tests = {}
	for month_lag in [-2,-1,0,1,2]:
		tests[month_lag] = data_setup.gen_ragged_dataset(X_test, variables, month_lag, missings, tmp)


	### model parameters
	n_features = X.shape[2]
	net = mv_net.MV_LSTM(n_features,n_timesteps, n_hidden, n_layers)
	criterion = torch.nn.L1Loss()
	optimizer = torch.optim.Adam(net.parameters(), lr=lr)
	
	# training
	net, train_loss = model_training.train_model(X, y, n_timesteps, net, criterion, optimizer, train_episodes, batch_size, decay)
	convergence_plot = plt.plot(train_loss)
	plt.xlabel("epoch")
	plt.ylabel("train loss")
	
	
	### evaluation
	# eval plot
	eval_plot = plt.plot(y_test, label="actual")
	for month_lag in [-2,-1,0,1,2]:
		plt.plot(model_training.gen_preds(tests[month_lag], net), label=f"{month_lag} months")
	plt.legend()
	
	# eval df
	tmp_eval = pd.DataFrame({
			"target_variable":target_variable, 
			"RMSE_2_back": np.sqrt(np.mean((model_training.gen_preds(tests[-2], net)[:-1]-y_test[:-1])**2)), # excluding last one, Q2 2020
			"RMSE_1_back": np.sqrt(np.mean((model_training.gen_preds(tests[-1], net)[:-1]-y_test[:-1])**2)),
			"RMSE_0_back": np.sqrt(np.mean((model_training.gen_preds(tests[0], net)[:-1]-y_test[:-1])**2)),
			"RMSE_1_ahead": np.sqrt(np.mean((model_training.gen_preds(tests[1], net)[:-1]-y_test[:-1])**2)),
			"RMSE_2_ahead": np.sqrt(np.mean((model_training.gen_preds(tests[2], net)[:-1]-y_test[:-1])**2)),
			"RMSE": np.mean([np.sqrt(np.mean((model_training.gen_preds(tests[-2], net)[:-1]-y_test[:-1])**2)),np.sqrt(np.mean((model_training.gen_preds(tests[-1], net)[:-1]-y_test[:-1])**2)),np.sqrt(np.mean((model_training.gen_preds(tests[0], net)[:-1]-y_test[:-1])**2)),np.sqrt(np.mean((model_training.gen_preds(tests[1], net)[:-1]-y_test[:-1])**2)),np.sqrt(np.mean((model_training.gen_preds(tests[2], net)[:-1]-y_test[:-1])**2))]),
			"MAE_2_back": np.abs(model_training.gen_preds(tests[-2], net)[:-1]-y_test[:-1]).mean(),
			"MAE_1_back": np.abs(model_training.gen_preds(tests[-1], net)[:-1]-y_test[:-1]).mean(),
			"MAE_0_back": np.abs(model_training.gen_preds(tests[0], net)[:-1]-y_test[:-1]).mean(),
			"MAE_1_ahead": np.abs(model_training.gen_preds(tests[1], net)[:-1]-y_test[:-1]).mean(),
			"MAE_2_ahead": np.abs(model_training.gen_preds(tests[2], net)[:-1]-y_test[:-1]).mean(),
			"MAE": np.mean([np.abs(model_training.gen_preds(tests[-2], net)[:-1]-y_test[:-1]).mean(),np.abs(model_training.gen_preds(tests[-1], net)[:-1]-y_test[:-1]).mean(),np.abs(model_training.gen_preds(tests[0], net)[:-1]-y_test[:-1]).mean(),np.abs(model_training.gen_preds(tests[1], net)[:-1]-y_test[:-1]).mean(),np.abs(model_training.gen_preds(tests[2], net)[:-1]-y_test[:-1]).mean()]),
			"q22020_2_back": model_training.gen_preds(tests[-2], net)[-1],
			"q22020_1_back": model_training.gen_preds(tests[-1], net)[-1],
			"q22020_0_back": model_training.gen_preds(tests[0], net)[-1],
			"q22020_1_ahead": model_training.gen_preds(tests[1], net)[-1],
			"q22020_2_ahead": model_training.gen_preds(tests[2], net)[-1],
			"variables":str(variables),
			"network":str(net).replace("\n",""),
			"n_timesteps":n_timesteps,
			"train_episodes":train_episodes,
			"batch_size":batch_size,
			"lr":lr,
			"decay":decay,
			"missings": missings,
			   }, 
	index=[0]
	)
	eval_df = eval_df.append(tmp_eval).reset_index(drop=True)
	
	return (convergence_plot, eval_plot, eval_df)