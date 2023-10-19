import torch
import torch.nn as nn
import numpy as np


class mc_dropout(torch.nn.Module):
	def __init__(self, p=0.0):
		super().__init__()
		self.p = p

	def forward(self, x):
		return torch.nn.functional.dropout(x, p=self.p, training=True, inplace=True)


class UnivariateKenNet(nn.Module):
	""" Combining MLE and Monte-Carlo Dropout for simultaneous al. and ep. UQ as proposed by
		Kendall et al.: https://proceedings.neurips.cc/paper_files/paper/2017/hash/2650d6089a6d640c5e85b2b88265dc2b-Abstract.html
	"""
	def __init__(self):
		super(UnivariateKenNet, self).__init__()

		self.n_mc_samples = 128

		self.hidden = nn.Sequential(
			nn.Linear(in_features=1, out_features=128),
			nn.Mish(),
			nn.Linear(in_features=128, out_features=128),
			nn.Mish(),	
		)

		self.mc_block = nn.Sequential(
			nn.Linear(in_features=128, out_features=128),
			mc_dropout(p=0.2),
			nn.Mish(),
			nn.Linear(in_features=128, out_features=128),
			mc_dropout(p=0.2),
			nn.Mish(),
			nn.Linear(in_features=128, out_features=2)
		)
		self.apply(self.init_weights)

	def forward(self, x):
		batch_size, _ = x.shape
		x = self.hidden(x)
		mc_x = x.repeat(self.n_mc_samples, 1)
		mc_x = self.mc_block(mc_x)
		mc_x = mc_x.view(self.n_mc_samples, batch_size, -1)

		mc_mu = torch.mean(mc_x, axis=0)
		mu, log_aleatoric = torch.split(mc_mu, split_size_or_sections=1, dim=-1)

		return mu, log_aleatoric

	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)

	def get_prediction(self, x):
		self.eval()

		batch_size, _ = x.shape
		x = self.hidden(x)
		mc_x = x.repeat(self.n_mc_samples, 1)
		mc_x = self.mc_block(mc_x)
		mc_x = mc_x.view(self.n_mc_samples, batch_size, -1)

		mc_mu, mc_var = torch.mean(mc_x, axis=0), torch.var(mc_x, axis=0)
		mu, log_aleatoric = torch.split(mc_mu, split_size_or_sections=1, dim=-1)
		epistemic, meta_log_aleatoric = torch.split(mc_var, split_size_or_sections=1, dim=-1)

		mu = mu.detach().numpy().squeeze()
		aleatoric = np.sqrt(np.exp(log_aleatoric.detach().numpy().squeeze()))
		epistemic = epistemic.detach().numpy().squeeze()

		return mu, aleatoric, epistemic, meta_log_aleatoric, None


class MultivariateKenNet(nn.Module):
	def __init__(self, p):
		super(MultivariateKenNet, self).__init__()

		self.n_mc_samples = 128
		self.p = p
		self.n_decomposit_units = int((1 + self.p) * self.p / 2)

		self.hidden = nn.Sequential(
			nn.Linear(in_features=1, out_features=128),
			nn.Tanh(),
			nn.Linear(in_features=128, out_features=128),
			nn.Tanh(),	
		)

		self.mc_block = nn.Sequential(
			nn.Linear(in_features=128, out_features=128),
			mc_dropout(p=0.2),
			nn.Tanh(),
			nn.Linear(in_features=128, out_features=128),
			mc_dropout(p=0.2),
			nn.Tanh(),
			nn.Linear(in_features=128, out_features=self.p + self.p**2)
		)
		self.apply(self.init_weights)
		self.evidence = torch.nn.Softplus()

	def forward(self, x):
		batch_size, _ = x.shape
		x = self.hidden(x)
		mc_x = x.repeat(self.n_mc_samples, 1)
		mc_x = self.mc_block(mc_x)
		mc_x = mc_x.view(self.n_mc_samples, batch_size, -1)
		mc_x = torch.mean(mc_x, dim=0)

		mu, L = mc_x[:, :self.p], mc_x[:, self.p:].reshape((batch_size, self.p, self.p))
		L = torch.tril(L, diagonal=-1) + torch.diag_embed(1e-2 + self.evidence(torch.diagonal(L, dim1=-2, dim2=-1)))

		return mu, L

	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)

	def get_prediction(self, x):
		self.eval()
		batch_size, _ = x.shape
		x = self.hidden(x)
		mc_x = x.repeat(self.n_mc_samples, 1)
		mc_x = self.mc_block(mc_x)
		mc_x = mc_x.view(self.n_mc_samples, batch_size, -1)

		mc_mu, mc_L = mc_x[:, :, :self.p], mc_x[:, :, self.p:].reshape((self.n_mc_samples, batch_size, self.p, self.p))
		mc_L = torch.tril(mc_L, diagonal=-1) + torch.diag_embed(1e-2 + self.evidence(torch.diagonal(mc_L, dim1=-2, dim2=-1)))
		L = torch.mean(mc_L, dim=0)
		
		mu = torch.mean(mc_mu, dim=0).detach().numpy().squeeze()
		aleatoric = torch.cholesky_solve(L, torch.eye(self.p)).detach().numpy()
		epistemic = self.batch_covariance(mc_mu, batch_size).detach().numpy()
		meta_aleatoric = 0.

		return mu, aleatoric, epistemic, meta_aleatoric, None

	def batch_covariance(self, mc_x, batch_size):
		mc_x = mc_x.view(batch_size, self.n_mc_samples, -1)
		covs = torch.zeros((batch_size, self.p, self.p))

		means = mc_x[:, :, ...].mean(axis=1)
		residuals = mc_x[:, :, ...] - means.unsqueeze(1)
		
		prod = torch.einsum('bijk, bikl -> bijl', residuals.unsqueeze(-1), residuals.unsqueeze(-2))
		bcov = prod.sum(axis=1) / (self.n_mc_samples - 1)
		covs[:, ...] = bcov

		return covs
