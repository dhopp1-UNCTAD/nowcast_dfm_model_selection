import pandas as pd
import random
import os
os.chdir("/home/danhopp/dhopp1/UNCTAD/nowcast_dfm_model_selection/lstm/")
import helper.run_model as run_model

# historical performance
eval_df = pd.read_csv("evaluation.csv")

# variable ranking
ranking = pd.read_csv("../variable_ranking/norm_variable_ranking.csv")

for i in range(10): # run 500 models with different variables
	print(f"run {i}/500")
	# parameters	
	target_variable = "x_world"
	#variables = ranking.loc[ranking.target_variable == target_variable,:].reset_index(drop=True).iloc[:100,:] # random assortment of top 100 variables
	#variables = list(variables.loc[pd.unique([random.randint(0,99) for x in range(random.randint(6,20))]), "variable"].values)
	variables = ['x_oecd', 'x_fr', 'adv_rti_us', 'x_br', 'export_orders_nl', 'bci_jp', 'x_vol_cn', 'manuf_orders_de', 'export_orders_fr', 'x_be', 'manuf_emp_fut_it', 'x_vol_world2', 'x_ru'] # best set for x_world
	
	#variables = ["bci_jp", "x_nl", "x_it", "x_br", "ipi_de", "ipi_ru", "rti_vol_fr", "constr_ca", "bci_nl", "export_orders_de", "fc_x_us"]
	n_timesteps = 12
	train_episodes = 200
	batch_size = 30
	lr = 1e-2
	decay = 0.98
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
