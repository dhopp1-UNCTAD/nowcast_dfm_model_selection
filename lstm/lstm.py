import matplotlib.pyplot as plt
import torch
import numpy as np
import os
os.chdir("/home/danhopp/dhopp1/UNCTAD/nowcast_dfm_model_selection/lstm/")

import helper.data_setup as data_setup
import helper.mv_net as mv_net
import helper.model_training as model_training

# setting up data
variables = ["bci_jp", "x_nl", "x_it", "x_br", "ipi_de", "ipi_ru", "rti_vol_fr", "constr_ca", "bci_nl", "export_orders_de", "fc_x_us"]
tmp = data_setup.gen_raw_data("x_world", variables)
n_timesteps = 12

train = tmp.loc[:179,:]
test = tmp.loc[180:,:]

train_dataset = data_setup.gen_dataset(train, "x_world", variables)
X, y = data_setup.gen_x_y(train_dataset, n_timesteps)

# filtering out missing ys, target is quarterly
mask = y != 0.0
y = y[mask]
X = X[mask,:,:]

test_dataset = data_setup.gen_dataset(test, "x_world", variables)
X_test, y_test = data_setup.gen_x_y(test_dataset, n_timesteps)

# filtering out missing ys, target is quarterly
mask = y_test != 0.0
y_test = y_test[mask]
X_test = X_test[mask,:,:]

# model parameters
n_features = X.shape[2]
mv_net = mv_net.MV_LSTM(n_features,n_timesteps)
criterion = torch.nn.L1Loss()
optimizer = torch.optim.Adam(mv_net.parameters(), lr=1e-1)
train_episodes = 200
batch_size = 16

# training
mv_net = model_training.train_model(X, y, n_timesteps, mv_net, criterion, optimizer, train_episodes, batch_size)

# predictions
preds = model_training.gen_preds(X, mv_net)

# ragged edges
X_ragged = np.array(X_test)
for obs in range(X_test.shape[0]):
	for var in range(len(variables)):
		for ragged in range(1, 3): # how many months delay before data
			X_ragged[obs,X_test.shape[1]-ragged,var] = 0.0 # setting to missing data

preds = model_training.gen_preds(X_ragged, mv_net)
plt.plot(y_test)
plt.plot(preds)
plt.title(np.abs(preds - y_test).mean())