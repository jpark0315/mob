import torch
import torch.nn as nn
import numpy as np 


class ForwardModel(nn.Module):
	def __init__(self, 
		input_shape, 
		hidden_size = 2048):
		super().__init__()


		self.nn= nn.Sequential(
			nn.Linear(input_shape, hidden_size), 
			nn.ReLU(),
			nn.Linear(hidden_size, hidden_size),
			nn.ReLU(),
			nn.Linear(hidden_size, 1))

	def forward(self, x):
		return self.nn(x)

class ConservativeObjectiveModel:
	def __init__(self, 
		forward_model, 
		 forward_model_lr=0.001, alpha=5.0,
		 alpha_lr=0.01, overestimation_limit=0.5,
		 particle_lr=0.05, particle_gradient_steps=50,
		 entropy_coefficient=0.9, noise_std=0.0):

		self.forward_model = forward_model
		self.forward_model_opt = torch.optim.Adam(self.forward_model.parameters(), lr = forward_model_lr)

		#self.log_alpha = torch.autograd.Variable(torch.log(torch.Tensor(alpha)), requires_grad = True)
		#self.alpha = self.log_alpha.exp()
		self.alpha = alpha 
		#self.alpha_optim = torch.optim.Adam([self.log_alpha], lr = alpha_lr)

		self.overestimation_limit = overestimation_limit
		self.particle_lr = particle_lr
		self.particle_gradient_steps = particle_gradient_steps
		self.entropy_coefficient = entropy_coefficient
		self.noise_std = noise_std

		self.train_step = 100
		self.batch_size = 64


	def optimize(self, x, steps, **kwargs):
		"""
		Args:
			x: Tensor, starting point for the optimizer that will be updated 
				using gradient descent 
			steps:
				number of gradient ascent 


		"""

		def gradient_step(xt):
			 xt.requires_grad = True 

			 loss = -self.forward_model(xt).mean()
			 loss.backward()

			 xt = xt.detach()- self.particle_lr * xt.grad
			 return xt 

		for i in range(steps):

			x = gradient_step(x)

		return x 


	def corrupt(self, x):
		return x + self.noise_std * torch.normal(0,1,x.shape)

	def naive_train_step(self, x, y):
		statistics = dict()

		d_pos = self.forward_model(x)
		#print(d_pos, y, d_pos.shape, y.shape)
		mse = torch.nn.MSELoss()(y, d_pos)
		statistics['mse'] = mse 

		self.forward_model_opt.zero_grad()
		mse.backward()
		self.forward_model_opt.step()
		return statistics

	def train_step_(self, x, y):
		"""Perform a training step of gradient descent on an ensemble
		using bootstrap weights for each model in the ensemble
		Args:
		x: tf.Tensor
			a batch of training inputs shaped like [batch_size, channels]
		y: tf.Tensor
			a batch of training labels shaped like [batch_size, 1]
		Returns:
		statistics: dict
			a dictionary that contains logging information
		"""
		x = self.corrupt(x)
		statistics = dict()

		#calculate the prediction error and accuracy of the model 
		d_pos = self.forward_model(x)
		mse = torch.nn.MSELoss()(y, d_pos)
		statistics['mse'] = mse 

		x_neg = self.optimize(x, self.particle_gradient_steps)

		d_neg = self.forward_model(x_neg)
		overestimation = d_neg - d_pos 
		statistics['overestimation'] = torch.clone(overestimation.mean().detach())

		#TODO: train alpha
		alpha_loss = self.alpha* self.overestimation_limit - self.alpha * overestimation
		statistics['alpha'] = self.alpha

		model_loss = mse+ self.alpha * overestimation.mean()

		self.forward_model_opt.zero_grad()
		model_loss.backward()
		self.forward_model_opt.step() 

		return statistics 



	def train(self, x, y, include_overestimation = True):

		for i in range(self.train_step):
			batch_idx = np.random.choice(x.shape[0], self.batch_size, replace = False)
			x_batch, y_batch= torch.FloatTensor(x[batch_idx]), torch.FloatTensor(y[batch_idx]).reshape(-1,1)


			if include_overestimation:
				statistics = self.train_step_(x_batch,y_batch)
			else:
				statistics = self.naive_train_step(x_batch, y_batch )
			print(statistics)


	










