import pandas as pd
import os
os.chdir("/home/danhopp/dhopp1/UNCTAD/nowcast_dfm_model_selection/lstm/")
import helper.run_model as run_model

# historical performance
eval_df = pd.read_csv("evaluation.csv")

# parameters
variables = ["bci_jp", "x_nl", "x_it", "x_br", "ipi_de", "ipi_ru", "rti_vol_fr", "constr_ca", "bci_nl", "export_orders_de", "fc_x_us"]
target_variable = "x_world"
n_timesteps = 12
train_episodes = 200
batch_size = 30
lr = 1e-2
decay = 0.96
missings = 0.0
n_hidden = 20
n_layers = 2

# running model evaluation
convergence_plot, eval_plot, eval_df = run_model.run_model(
		variables = variables, 
		target_variable = target_variable, 
		n_timesteps = n_timesteps, 
		train_episodes = train_episodes, 
		batch_size = batch_size, 
		lr = lr, 
		decay = decay, 
		missings = missings, 
		n_hidden = n_hidden, 
		n_layers = n_layers, 
		eval_df = eval_df
		)

eval_df.to_csv("evaluation.csv", index=False)
