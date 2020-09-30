# import matplotlib.pyplot as plt
import torch
import os
os.chdir("/home/danhopp/dhopp1/UNCTAD/nowcast_dfm_model_selection/lstm/")

import helper.data_setup as data_setup
import helper.mv_net as mv_net
import helper.model_training as model_training

# setting up data
variables = ["bci_jp", "x_nl", "x_it", "x_br", "ipi_de", "ipi_ru", "rti_vol_fr", "constr_ca", "bci_nl", "export_orders_de"]
data = data_setup.gen_raw_data("x_world", variables)
dataset = data_setup.gen_dataset(data, "x_world", variables) # incorporate here setting the ragged ends to -1 for missing data
n_timesteps = 36
X, y = data_setup.gen_x_y(dataset, n_timesteps)

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