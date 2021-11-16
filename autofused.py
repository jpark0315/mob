import torch
import torch.nn as nn
import numpy as np 

#ok, so pretrain the vae first and then 
class Encoder(nn.Module):
	def __init__(self, input_shape, latent_size = 3, hidden_size = 128):
		super().__init__()

		self.layers = nn.Sequential(
			nn.Linear(input_shape, hidden_size),
			nn.ReLU(), 
			nn.Linear(hidden_size, latent_size *2 ))
		self.latent_size = latent_size
	def forward(self, x):
		return self.layers(x)

	def get_params(self,x, return_log = False):
		
		mean, log_std = torch.split(self(x),self.latent_size, dim = -1)
		if return_log:
			return mean, log_std
		else:
			return mean, log_std.exp()

	def get_distribution(self, x):
		mean, log_std = self.get_params(x)
		assert mean.shape == log_std.shape 
		#cov = torch.bmm(log_std.unsqueeze(2), log_std.unsqueeze(1))
		#return torch.distributions.MultivariateNormal(mean, cov)
		#return torch.distributions.MultivariateNormal(mean, torch.diag_embed(log_std))
		return torch.distributions.Normal(mean, log_std)


class Decoder(nn.Module):
	def __init__(self,input_shape, latent_size = 3,hidden_size = 128):
		super().__init__()

		self.layers = nn.Sequential(
			nn.Linear(latent_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size,input_shape* 2)
			)
		self.input_shape = input_shape

	def forward(self, x):
		return self.layers(x)
	def get_params(self,x, return_log = False):
		
		mean, log_std = torch.split(self(x),self.input_shape, dim = -1)
		if return_log:
			return mean, log_std
		else:
			return mean, log_std.exp()

	def get_distribution(self, x):
		mean, std = self.get_params(x)
		assert mean.shape == std.shape 
		#return torch.distributions.MultivariateNormal(mean, torch.bmm(log_std.unsqueeze(2), log_std.unsqueeze(1)) )
		return torch.distributions.Normal(mean, std)

		#return torch.distributions.MultivariateNormal(mean, torch.diag_embed(std))
		



class WeightedVAE:
	def __init__(self, input_shape, lr = 1e-3):
		self.encoder = Encoder(input_shape)
		self.decoder = Decoder(input_shape)

		params = list(self.encoder.parameters()) + list(self.decoder.parameters())

		self.optim = torch.optim.Adam(params, lr=lr)
		self.beta = 1

	def train_step(self, x, w):

		statistics = dict() 
		
		dz = self.encoder.get_distribution(x)
		z = dz.rsample()
		dx = self.decoder.get_distribution(z)

		nll = -dx.log_prob(x).sum(1)
		#kl = torch.distributions.kl_divergence(self.encoder.get_distribution(x), 
		#	torch.distributions.MultivariateNormal(torch.zeros(z.shape), torch.diag_embed(torch.ones(z.shape))))
		kl = torch.distributions.kl_divergence(self.encoder.get_distribution(x), 
			torch.distributions.Normal(torch.zeros(z.shape), torch.ones(z.shape))).sum(1)
		#print(w.shape, kl.shape, nll.shape )
		total_loss = (w*(self.beta * kl + nll)).mean()
		#total_loss = nll.mean()
		self.optim.zero_grad()
		total_loss.backward()
		self.optim.step() 

		statistics['nll'] = nll.mean().detach() 
		statistics['kl'] = kl.mean().detach() 
		return statistics

	def sample(self, n=100, check_log_prob = False):
		z = torch.distributions.Normal(torch.zeros((n,self.encoder.latent_size)), torch.ones(n,self.encoder.latent_size)).sample()
		sample = self.decoder.get_distribution(z).sample()
		return sample 



	def train(self, x, w, epochs = 10000, batch_size = 128, verbal = True): 

		for i in range(epochs):
			idx= np.random.permutation(len(x))[:batch_size]
			x_batch = torch.FloatTensor(x[idx])
			w_batch = torch.FloatTensor(w[idx])
			statistics = self.train_step(x_batch, w_batch)	
			if verbal: print(statistics)

	def initiate(self, x):
		w = torch.ones(len(x))
		self.train(x, w)





class ForwardModel(nn.Module):
	def __init__(self, task, 
		input_shape, 
		embedding_size = 50,
		hidden_size = 128, num_layers = 1):
		super().__init__()

		layers = nn.Sequential(
			nn.Linear(input_shape,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 2))

	def forward(self, x):
		return self.layers(x)

	def get_params(self,x, return_log = False):

		mean, log_std = torch.split(self(x),2, axis = -1)
		if return_log:
			return mean, log_std
		else:
			return mean, log_std.exp()

	def get_distribution(self, x):
		return torch.distributions.Normal(self.get_params(x))


class Ensemble:
	def __init__(self):
		self.forward

		self.bootstraps=1


	def train_step(self, x, y, b, w):
		statistics = dict()
		for i in range(self.bootstraps):
			fm = self.forward_models[i]


#wait so you are updating the model with the updated x and an old y? 