import torch
import numpy as np
from scipy.stats import norm, multivariate_normal, invwishart


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_predicted_cdf(residuals: np.ndarray, sigma: np.ndarray):
    """ Using residuals, generates confidence scores by comparing
        to the standard Gaussian, scaled by predicted standard deviations.
    """
    alpha = np.linspace(start=1.0, stop=0, num=10)
    observed_confidence_p = np.zeros((len(residuals), len(alpha)))

    # generate quantiles for the standard Gaussian
    std_quantiles = norm.ppf(alpha)

    # weight residuals with predicted standard deviations
    weighted_residuals = residuals / sigma

    # for each quantile, check whether the weighted residual lies within
    observed_confidence_p = np.less_equal(np.expand_dims(weighted_residuals, axis=-1), std_quantiles)

    # get sample cdf by summing the number of quantiles the sample error lies inside of
    pcdf = observed_confidence_p.mean(axis=-1)
    return pcdf


class UnivariateDummyData:
	def __init__(self, N, X_range=(0, 10.0)):
		epsilon = 0.3 * np.random.normal(loc=0.0, scale=1.0, size=N)
		self.X = np.linspace(*X_range, num=N)
		self.Y = self.X * np.sin(self.X) + self.X * epsilon + epsilon

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		x = torch.Tensor(np.expand_dims(self.X[idx], axis=0))
		y = torch.Tensor(np.expand_dims(self.Y[idx], axis=0))
		return x, y


class MultivariateDummyData:
	def __init__(self, N, X_range=(0, 10.0)):
		epsilon = np.random.multivariate_normal(np.array([0., 0.]), np.array([[0.8, -0.3], [-0.3, 0.8]]), size=N)

		self.X = np.linspace(*X_range, num=N)
		self.Y = self.X * np.sin(self.X) + self.X * 0.3 * epsilon[:, 0] + epsilon[:, 0]
		self.Z = self.X * np.cos(self.X) + self.X * 0.3 * epsilon[:, 1] + epsilon[:, 1]

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		x = torch.Tensor(np.expand_dims(self.X[idx], axis=0))
		y = torch.Tensor(np.expand_dims(self.Y[idx], axis=0))
		z = torch.Tensor(np.expand_dims(self.Z[idx], axis=0))
		return x, y, z
