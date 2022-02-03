import torch
import torch.nn as nn
import numpy as np 
import copy 
LOG_STD_MAX = 2
LOG_STD_MIN = -20
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
		log_std = torch.clamp(
			log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
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
		log_std = torch.clamp(
			log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
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

	def train_step(self, x, w, log_prob= True):

		statistics = dict() 
		
		dz = self.encoder.get_distribution(x)
		z = dz.rsample()
		dx = self.decoder.get_distribution(z)


		if log_prob:
			nll = -dx.log_prob(x).sum(1)
		else:
			nll = torch.nn.MSELoss()(dx.rsample(), x) 
		#kl = torch.distributions.kl_divergence(self.encoder.get_distribution(x), 
		#	torch.distributions.MultivariateNormal(torch.zeros(z.shape), torch.diag_embed(torch.ones(z.shape))))
		kl = torch.distributions.kl_divergence(self.encoder.get_distribution(x), 
			torch.distributions.Normal(torch.zeros(z.shape), torch.ones(z.shape))).sum(1)
		#print(w.shape, kl.shape, nll.shape )
		#total_loss = (w*(self.beta * kl + nll)).mean()
		nll = nll * w 
		total_loss = (self.beta * kl + nll).mean()
		#total_loss = nll.mean()
		self.optim.zero_grad()
		total_loss.backward()
		self.optim.step() 

		statistics['nll'] = nll.mean().detach() 
		statistics['kl'] = kl.mean().detach() 
		return statistics

	def sample(self, x = None, n=100, check_log_prob = False, return_dist = False):
		if x is not None:
			with torch.no_grad():
				dz = self.encoder.get_distribution(x)
				z = dz.sample()
				sample = self.decoder.get_distribution(z).sample() 
		else:
			z = torch.distributions.Normal(torch.zeros((n,self.encoder.latent_size)), torch.ones(n,self.encoder.latent_size)).sample()
			dist = self.decoder.get_distribution(z)
			sample = dist.sample()

		return dist if return_dist else sample 



	def train(self, x, w, epochs = 10000, batch_size = 128, verbal = True): 

		for i in range(epochs):
			idx= np.random.permutation(len(x))[:batch_size]
			x_batch = torch.FloatTensor(x[idx])
			w_batch = torch.FloatTensor(w[idx]).reshape(-1,1)
			statistics = self.train_step(x_batch, w_batch)	
			if verbal: print(statistics)

	def initiate(self, x):
		w = torch.ones(len(x)).reshape(-1,1)
		self.train(x, w)





class ForwardModel(nn.Module):
	def __init__(self, 
		input_shape, 
		embedding_size = 50,
		hidden_size = 128, num_layers = 1, lr=1e-3):
		super().__init__()

		self.layers = nn.Sequential(
			nn.Linear(input_shape,hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 2))

		self.entropy_coef = 5
		self.optim = torch.optim.Adam(self.parameters(), lr = lr)

	def forward(self, x):
		if not isinstance(x, torch.Tensor):
			x = torch.FloatTensor(x)
		return self.layers(x)

	def get_params(self,x, return_log = False):

		mean, log_std = torch.split(self(x),1, dim=-1)
		log_std = torch.clamp(
			log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
		if return_log:
			return mean, log_std
		else:
			return mean, log_std.exp()

	def get_distribution(self, x):
		mean, std = self.get_params(x)
		return torch.distributions.Normal(mean, std)

	def train_step(self, x, y, w, log_prob= True, include_entropy= False):
		statistics = {}
		#print(include_entropy)
		d = self.get_distribution(x)
		if log_prob:
			nll = -d.log_prob(y)

		else:
			pred = d.rsample()
			nll = torch.nn.MSELoss()(pred, y)

		if include_entropy:
			#nll = nll 
			nll = nll - self.entropy_coef* d.entropy().mean() 

		weighted_nll = nll * w

		self.optim.zero_grad()
		weighted_nll.mean().backward()
		self.optim.step()

		statistics['nll'] = nll.mean().detach()
		statistics['weighted_nll'] = weighted_nll.mean().detach() 
		
		statistics['entorpy'] = d.entropy().mean().detach() 
		return statistics

	def train(self, x, y,w, train_step = 1000, batch_size = 128, verbal = True):

		for i in range(3000):

			idx= np.random.permutation(len(x))[:batch_size]
			x_batch = torch.FloatTensor(x[idx])
			w_batch = torch.FloatTensor(w[idx])
			y_batch = torch.FloatTensor(y[idx].reshape(-1,1))
			statistics = self.train_step(x_batch, y_batch, w_batch)

			if verbal: print(statistics)

	def initiate(self, x,y):
		w = torch.ones(len(x)).reshape(-1,1 )
		self.train(x, y, w)




class CBAES:

	def __init__(self, vae = None, fm=None):
		self.vae = WeightedVAE().initiate(x,y) if vae is None else vae 
		self.fm = ForwardModel().initiate(x,y) if fm is None else fm 
		self.initial_vae = copy.deepcopy(self.vae)

		self.vae_train_epochs = 1000

	def get_data(self,x, weight_from_fm = False):
		"""
		Args: 
			x-> training batch of x
		Returns: 
			gamma-> Qth percentile in terms of score of the generated samples
			vae_weight-> previous weight * cdf weights

		"""
		if not isinstance(x,torch.Tensor):
			x = torch.FloatTensor(x)

		#sample new x and get its score 
		vaedata = self.vae.sample(x)
		dist = self.fm.get_distribution(vaedata)

		#How does the newly created x fare in logprob vs the old vae? 
		vae_dist = self.vae.sample(n = len(vaedata), return_dist=True)
		initial_vae_dist = self.initial_vae.sample(n = len(vaedata), return_dist = True)
		logprob, initial_logprob = vae_dist.log_prob(vaedata).mean().detach().item(), initial_vae_dist.log_prob(vaedata).mean().detach().item()

		if weight_from_fm: 
			score = dist.sample()
			print(score.mean())
			weight = (score-min(score))/(max(score)-min(score))

		else:

			top_data, gamma = self.return_topk(dist.sample(), vaedata)
			cdf_weights = self.get_cdf_weights(dist, gamma)

			weight = cdf_weights 


		return vaedata, weight, logprob, initial_logprob

	def train(self, x, y,coef, verbal = False, weight_from_fm = True, verbose = True): 

		new_x, weights, lp,ilp = self.get_data(x, weight_from_fm =weight_from_fm)
		assert len(weights) == len(new_x)
		assert new_x.shape == x.shape 

		if verbose:
			self.evaluate(coef, x = x )
			print('lp,ilp:' ,lp, ilp)

		self.vae.train(new_x, weights, epochs = self.vae_train_epochs, verbal = verbal)


	def evaluate(self, coef, x = None):
		#evaluate vae and fm 
		
		if x is not None:
			x = torch.Tensor(x)
		sample = self.vae.sample(x = x).numpy() 
		
		if isinstance(coef, np.ndarray):
			true_score = sample @ coef
		else:
			true_score = coef.predict(coef.denormalize_x(sample))
		fake_score_dist = self.fm.get_distribution(sample )
		fake_score = fake_score_dist.sample()
		
		top_data, gamma, top_indices = self.return_topk(fake_score,sample, percentile = 0.05)
		
		print('Models Likelihood of true score', fake_score_dist.log_prob(torch.Tensor(true_score)).mean().detach().item())
		print('Average Models score vs true score:',true_score.mean(), fake_score.mean().item())
		print('Top Percentile Models score vs true score:', true_score[top_indices].mean(), fake_score[top_indices].mean().item())
		print('Top true score, top fake score', true_score.max(), fake_score.max())
	@staticmethod
	def return_topk(sample,data, percentile = 0.1):
		k = len(sample) * percentile
		top = torch.topk(sample.reshape(-1),int(k))
		return data[top.indices], top.values[-1].item(), top.indices 

	@staticmethod 
	def get_cdf_weights(dist, gamma):
		"""
		gamma is float 
		"""
		weights = 1-dist.cdf(torch.Tensor([gamma]))
		return weights








"""
experiment ideas:
ensembling f 

generalization: 
vae representation learning for training+generated samples, 
vae representation learning but with a shared encoder 
vae encoder trained to minimize training loss as well? 
regularization technique from papers 
DANN 

"""