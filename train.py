import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import get_device, UnivariateDummyData, get_predicted_cdf

from evidential_regression.networks import UnivariateDerNet
from evidential_regression.losses import UnivariateEvidentialRegressionLoss

from mle_mc_dropout.networks import UnivariateKenNet
from mle_mc_dropout.losses import UnivariateL1Loss, UnivariateL2Loss, BetaNLLLoss

# plot settings
plt.rcParams.update(
    {
    	'font.size': 12,
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)


if __name__ == '__main__':
	EPOCHS = 120

	in_lower = -2.0
	in_upper = 10.0

	train_data = UnivariateDummyData(N=2000, X_range=(in_lower, in_upper))
	test_data  = UnivariateDummyData(N=100, X_range=(-10.0, 20.0))

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

	optimizer_params = {
	    "lr": 1e-03,
	    "betas": (0.9, 0.999),
	    "eps": 1e-8,
	    "weight_decay": 1e-2,
	    "amsgrad": False}

	# net = UnivariateDerNet()
	# criterion = UnivariateEvidentialRegressionLoss()

	net = UnivariateKenNet()
	criterion = BetaNLLLoss()

	device = get_device()
	optimizer = torch.optim.AdamW(net.parameters(), **optimizer_params)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer_params["lr"], steps_per_epoch=len(train_loader), epochs=EPOCHS)

	losses = []
	t = tqdm.trange(EPOCHS)
	for i in t:
		net.train()
		for (x_batch, y_batch) in train_loader:
			inputs = x_batch.to(device)
			labels = y_batch.to(device)

			optimizer.zero_grad()
			outs = net(inputs)
			loss = criterion(labels, *outs)

			loss.backward()
			optimizer.step()
			scheduler.step()

		t.set_description(f"val. loss:{loss.detach().numpy():.2f} ")
		t.refresh()
		losses += [loss.detach().numpy()]

		net.eval()

		mu, aleatoric, epistemic, meta_aleatoric, output_params = net.get_prediction(torch.Tensor(np.expand_dims(test_data.X, axis=1)))

	""" Visualizing the experiment
	"""
	plt.scatter(test_data.X, test_data.Y, marker="+", color="black")

	# plot in-distribution limits
	plt.plot([in_lower, in_lower], [-20, 20], color="grey", linestyle="dotted")
	plt.plot([in_upper, in_upper], [-20, 20], color="grey", linestyle="dotted")

	# plot aleatoric and epistemic uncertainty on top of each other
	plt.fill_between(test_data.X, (mu - aleatoric).squeeze(), (mu + aleatoric).squeeze(), color="#8a2be280", linewidth=0, label="$\mathbb{E}[\sigma^2]$ (aleatoric)")
	plt.fill_between(test_data.X, (mu - (epistemic + aleatoric)).squeeze(), (mu - aleatoric).squeeze(), color="#3cb37180", linewidth=0, label="$\mathrm{Var}[\mu]$ (epistemic)")
	plt.fill_between(test_data.X, (mu + (epistemic + aleatoric)).squeeze(), (mu + aleatoric).squeeze(), color="#3cb37180", linewidth=0)
	# plt.plot(test_data.X, (mu - meta_aleatoric), color="#80808080", linestyle="--")
	# plt.plot(test_data.X, (mu + meta_aleatoric), color="#80808080", linestyle="--")

	# plot predicted function
	plt.plot(test_data.X, mu, color="black", label="$\hat \mu$")

	# plot ground truth function
	plt.plot(test_data.X, test_data.X * np.sin(test_data.X), color="#88888880", label="true mean")
	plt.plot(test_data.X, test_data.X * np.sin(test_data.X) + (test_data.X * 0.3 + 0.3), color="#88888880", linestyle="--", label="true variance")
	plt.plot(test_data.X, test_data.X * np.sin(test_data.X) - (test_data.X * 0.3 + 0.3), color="#88888880", linestyle="--")

	plt.legend()
	plt.ylim([-20, 20])
	
	fig = plt.gcf()
	plt.locator_params(axis="y", nbins=5)
	plt.tight_layout()
	plt.show()

	""" Creating and plotting calibration plots
	"""
	mask = np.logical_and(test_data.X > in_lower, test_data.X < in_upper)
	in_Y = test_data.Y[mask]
	in_mu = mu[mask]
	in_al = aleatoric[mask]
	pcdf = get_predicted_cdf(residuals=in_mu - in_Y, sigma=in_al)
	
	pcal = []
	for p in np.arange(0.1, 1.1, 0.1):
		pcal += [np.sum(pcdf <= p, axis=0) / max(1, len(pcdf))]
	plt.plot(np.arange(0.1, 1.1, 0.1), np.arange(0.1, 1.1, 0.1), color='black', linestyle='--')
	plt.plot(np.arange(0.1, 1.1, 0.1), pcal)
	plt.title(r'Calibration plot of $\mathbb{E} [\Sigma]$')

	plt.locator_params(axis='both', nbins=3) 
	plt.xticks([0.1, 0.5, 1.0], [0.1, 0.5, 1.0])
	plt.yticks([0.1, 0.5, 1.0], [0.1, 0.5, 1.0])
	plt.show()

	""" Plotting loss curve
	"""
	plt.plot(losses)
	plt.show()

	plt.clf()
