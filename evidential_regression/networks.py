import torch
import torch.nn as nn
import numpy as np

from .layers import DenseInverseGamma, DenseInverseWishart


class UnivariateDerNet(nn.Module):
	def __init__(self):
		super(UnivariateDerNet, self).__init__()

		self.hidden = nn.Sequential(
			nn.Linear(in_features=1, out_features=128),
			nn.Mish(),
			nn.Linear(in_features=128, out_features=128),
			nn.Mish(),
			nn.Linear(in_features=128, out_features=128),
			nn.Mish(),
			nn.Linear(in_features=128, out_features=128),
			DenseInverseGamma(in_features=128, units=1)
		)
		self.apply(self.init_weights)

	def forward(self, x):
		gamma, nu, alpha, beta = self.hidden(x)

		return gamma, nu, alpha, beta

	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)

	def get_prediction(self, x):
		self.eval()

		gamma, nu, alpha, beta = self.hidden(x)

		gamma = gamma.detach().cpu().numpy().squeeze()
		nu = nu.detach().cpu().numpy().squeeze()
		alpha = alpha.detach().cpu().numpy().squeeze()
		beta = beta.detach().cpu().numpy().squeeze()

		aleatoric = np.sqrt(beta * np.reciprocal(alpha - 1 + 1e-8))
		epistemic = np.sqrt(beta * np.reciprocal((nu * (alpha - 1)) + 1e-8))
		meta_aleatoric = np.sqrt(beta**2 / ((alpha - 1)**2 * (alpha - 2 + 1e-6)))

		return gamma, aleatoric, epistemic, meta_aleatoric, {"nu": nu, "alpha": alpha, "beta": beta}


class MultivariateDerNet(nn.Module):
	def __init__(self, p):
		super(MultivariateDerNet, self).__init__()
		self.p = p

		self.hidden = nn.Sequential(
			nn.Linear(in_features=1, out_features=128),
			# nn.ReLU6(),
			nn.Tanh(),
			# nn.Mish(),
			nn.Linear(in_features=128, out_features=128),
			# nn.ReLU6(),
			nn.Tanh(),
			# nn.Mish(),
			nn.Linear(in_features=128, out_features=128),
			# nn.ReLU6(),
			nn.Tanh(),
			# nn.Mish(),
			nn.Linear(in_features=128, out_features=128),
			DenseInverseWishart(in_features=128, p=self.p)
		)
		self.apply(self.init_weights)

	def forward(self, x):
		mu, nu, kappa, L = self.hidden(x)

		return mu, nu, kappa, L

	def init_weights(self, m):
		if isinstance(m, nn.Linear):
			torch.nn.init.xavier_uniform_(m.weight)

	def get_prediction(self, x):
		self.eval()

		mu, nu, kappa, L = self.hidden(x)

		mu = mu.detach().cpu().numpy().squeeze()
		nu = nu.detach().cpu().numpy().squeeze(axis=1)
		kappa = kappa.detach().cpu().numpy().squeeze()
		L = L.detach().cpu().numpy()

		sum_of_pairwise_deviation_products = np.einsum('bik, bkl -> bil', L, np.transpose(L, (0, -1, -2)))
		aleatoric = np.reciprocal(nu[:, None, None] - self.p - 1 + 1e-8) * sum_of_pairwise_deviation_products
		epistemic = np.reciprocal(nu[:, None, None] + 1e-8) * aleatoric
		meta_aleatoric = np.zeros_like(aleatoric)
		for i, j in zip(range(self.p), range(self.p)):
			meta_aleatoric[:, i, j] = (nu - self.p + 1) * aleatoric[:, i, j] + (nu - self.p - 1) * aleatoric[:, i, i] * aleatoric[:, j, j]
			meta_aleatoric[:, i, j] /= (nu - self.p) * (nu - self.p - 1)**2 * (nu - self.p - 3)

		return mu, aleatoric, epistemic, meta_aleatoric, {"nu": nu, "kappa": kappa, "L": L}
