import pandas as pd
import random
import os
os.chdir("/home/danhopp/dhopp1/UNCTAD/nowcast_dfm_model_selection/lstm/")
import helper.run_model as run_model

# historical performance
eval_df = pd.read_csv("evaluation.csv")

# variable ranking
ranking = pd.read_csv("../variable_ranking/norm_variable_ranking.csv")

for target_variable in ranking.target_variable.unique():
	for i in range(500): # run 500 models with different variables
		print(f"{target_variable}: run {i}")
		# parameters	
		variables = ranking.loc[ranking.target_variable == target_variable,:].reset_index(drop=True).iloc[:i,:] # random assortment of top 100 variables
		variables = list(variables.loc[pd.unique([random.randint(0,99) for x in range(random.randint(5,30))]), "variable"].values)
		#variables = ['x_oecd', 'x_fr', 'adv_rti_us', 'x_br', 'export_orders_nl', 'bci_jp', 'x_vol_cn', 'manuf_orders_de', 'export_orders_fr', 'x_be', 'manuf_emp_fut_it', 'x_vol_world2', 'x_ru'] # best set for x_world
		#variables = ["bci_jp", "x_nl", "x_it", "x_br", "ipi_de", "ipi_ru", "rti_vol_fr", "constr_ca", "bci_nl", "export_orders_de", "fc_x_us"] # original x_world testing dataset
		#variables = ['x_nl', 'ipi_jp', 'rti_val_fr', 'rti_val_es', 'cci_br', 'bci_nl', 'export_orders_it', 'export_orders_uk', 'manuf_orders_it', 'ipi_eu27', 'x_servs_us', 'manuf_orders_us_2', 'x_servs_sg', 'p_manuf', 'x_world', 'x_vol_ez', 'x_servs_world', 'manuf_emp_fut_it', 'fc_gdp_us', 'fc_x_us', 'fc_x_de'] # DFM variables for services
		
		n_timesteps = 12
		train_episodes = 200
		batch_size = 30
		lr = 1e-2
		decay = 0.98
		missings = 0.0
		n_hidden = 20
		n_layers = 2
		dropout = 0
		
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
				dropout = dropout,
				eval_df = eval_df
				)
		
		eval_df.to_csv("evaluation.csv", index=False)
