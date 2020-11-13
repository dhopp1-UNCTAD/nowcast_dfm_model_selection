import torch

def train_model(X, y, n_timesteps, mv_net, criterion, optimizer, train_episodes, batch_size):
	train_loss = [] # for plotting train loss
	my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.96) # for learning rate decay
	mv_net.train()
	for t in range(train_episodes):
	    for b in range(0,len(X),batch_size):
	        inpt = X[b:b+batch_size,:,:]
	        target = y[b:b+batch_size]    
	        
	        x_batch = torch.tensor(inpt,dtype=torch.float32)
	        y_batch = torch.tensor(target,dtype=torch.float32)
	    
	        mv_net.init_hidden(x_batch.size(0))
	        output = mv_net(x_batch) 
	        loss = criterion(output.view(-1), y_batch)  
	        
	        loss.backward()
	        my_lr_scheduler.step() # optimizer.step() etc. if don't do learning rate decay
	        my_lr_scheduler.zero_grad()
			
	    print('step : ' , t , 'loss : ' , loss.item())
	    train_loss.append(loss.item())
	return (mv_net, train_loss)

def gen_preds(X, net):
	inpt = torch.tensor(X,dtype=torch.float32)    
	net.init_hidden(inpt.size(0))
	preds = net(inpt).view(-1).detach().numpy()
	return preds