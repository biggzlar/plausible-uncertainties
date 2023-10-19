import tqdm
import torch
import numpy as np
import mpl_toolkits.mplot3d.art3d as art3d
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt

from utils import get_device, MultivariateDummyData, get_predicted_cdf

from evidential_regression.networks import MultivariateDerNet
from evidential_regression.losses import MultivariateEvidentialRegressionLoss

from mle_mc_dropout.networks import MultivariateKenNet
from mle_mc_dropout.losses import MultivariateGaussianNLL

# plot settings
plt.rcParams.update(
    {
    	'font.size': 12,
        'text.usetex': False,
        'font.family': 'stixgeneral',
        'mathtext.fontset': 'stix',
    }
)


def confidence_ellipse(x, y, z, cov, ax, n_std=1.0, **kwargs):
	""" Method to draw 2d ellipses in 3d plots.
	"""
	lambda_, v = np.linalg.eig(cov)
	lambda_ = np.minimum(np.sqrt(lambda_), [10.])
	ellipse = Ellipse((y, z), width=lambda_[0] * 3 * 2, height=lambda_[1] * 3 * 2, 
		angle=np.rad2deg(np.arccos(v[0, 0])), **kwargs)

	ax.add_patch(ellipse)
	art3d.pathpatch_2d_to_3d(ellipse, z=x, zdir="x")
	return


if __name__ == '__main__':
	cmap = plt.cm.bone_r

	EPOCHS = 200

	in_lower = -10.0
	in_upper = 4.0
	out_lower = -20.0
	out_upper = 10.0

	train_data = MultivariateDummyData(N=8000, X_range=(in_lower, in_upper))
	test_data  = MultivariateDummyData(N=200, X_range=(out_lower, out_upper))

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)
	test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)
	test_YZ = np.concatenate([np.expand_dims(test_data.Y, axis=1), np.expand_dims(test_data.Z, axis=1)], axis=-1)

	optimizer_params = {
		"lr": 1e-03,
		"betas": (0.9, 0.999),
		"eps": 1e-8,
		"weight_decay": 1e-2,
		"amsgrad": False}

	# choice of model/method
	net = MultivariateDerNet(p=2)
	criterion = MultivariateEvidentialRegressionLoss()

	# net = MultivariateKenNet(p=2)
	# criterion = MultivariateGaussianNLL()

	device = get_device()
	optimizer = torch.optim.AdamW(net.parameters(), **optimizer_params)
	scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=optimizer_params["lr"], steps_per_epoch=len(train_loader), epochs=EPOCHS)

	losses = []
	t = tqdm.trange(EPOCHS)
	for i in t:
		net.train()
		for (x_batch, y_batch, z_batch) in train_loader:
			inputs = x_batch.to(device)
			labels = torch.concat([y_batch, z_batch], dim=-1).to(device)

			optimizer.zero_grad()
			outs = net(inputs)
			loss = criterion(labels, *outs)

			loss.backward()
			optimizer.step()
			scheduler.step()

		net.eval()

		mu, aleatoric, epistemic, meta_aleatoric, output_params = net.get_prediction(torch.Tensor(np.expand_dims(test_data.X, axis=1)))
		
		t.set_description(f"val. loss: {loss.detach().numpy():.2f}")
		t.refresh()
		losses += [loss.detach().numpy()]

	""" Visualizing the experiment
	"""
	ax = plt.axes(projection='3d')
	ax.scatter3D(test_data.X, test_data.Y, test_data.Z, marker="+", color="black")

	# plot in-distribution limits
	rect0 = Rectangle((-20, -20), 40, 40, fill=False, hatch='X')
	ax.add_patch(rect0)
	art3d.pathpatch_2d_to_3d(rect0, z=in_lower, zdir="x")
	rect1 = Rectangle((-20, -20), 40, 40, fill=False, hatch='X')
	ax.add_patch(rect1)
	art3d.pathpatch_2d_to_3d(rect1, z=in_upper, zdir="x")

	# plot aleatoric (and epistemic) uncertainty
	for j in range(len(test_data)):
		confidence_ellipse(test_data.X[j], mu[j, 0], mu[j, 1], aleatoric[j], ax, 
			facecolor=cmap(j / len(test_data)), edgecolor=None, alpha=0.3)

	# plot predicted function
	plt.plot(test_data.X, mu[:, 0], mu[:, 1], color="black", label="$\hat \mu$")

	# plot ground truth function
	plt.plot(test_data.X, test_data.X * np.sin(test_data.X), test_data.X * np.cos(test_data.X), color="#88888880", label="true mean")

	# # plot ground truth aleatoric uncertainty
	# for x in test_data.X:
	# 	confidence_ellipse(x, x * np.sin(x), x * np.cos(x), x * 0.3 * np.array([[0.8, -0.3], [-0.3, 0.8]]), ax,
	# 		fill=None, edgecolor='black', linestyle='--')

	fig = plt.gcf()
	ax.set_xlim(out_lower, out_upper)
	ax.set_ylim(-20, 20)
	ax.set_zlim(-20, 20)
	
	ax.locator_params(axis="x", nbins=5)
	ax.locator_params(axis="y", nbins=5)
	ax.locator_params(axis="z", nbins=5)
	plt.tight_layout()
	# plt.legend()
	import pickle
	pickle.dump(fig, open('mv_der.fig.pickle', 'wb'))
	plt.show()
	
	""" Creating and plotting calibration plots
	"""
	in_YZ = test_YZ[np.logical_and(test_data.X > in_lower, test_data.X < in_upper)]
	in_mu = mu[np.logical_and(test_data.X > in_lower, test_data.X < in_upper)]
	in_al = aleatoric[np.logical_and(test_data.X > in_lower, test_data.X < in_upper)]
	pcdf = get_predicted_cdf(residuals=in_mu - in_YZ, sigma=np.diagonal(in_al, axis1=-2, axis2=-1))
	
	pcal = []
	for p in np.arange(0.1, 1.1, 0.1):
		pcal += [np.sum(pcdf <= p, axis=0) / max(1, len(pcdf))]
	plt.plot(np.arange(0.1, 1.1, 0.1), np.arange(0.1, 1.1, 0.1), color='black', linestyle='--')
	plt.plot(np.arange(0.1, 1.1, 0.1), pcal)
	plt.title(r'Calibration plot of diagonal elements of $\mathbb{E} [\Sigma]$')

	plt.locator_params(axis='both', nbins=3) 
	plt.xticks([0.1, 0.5, 1.0], [0.1, 0.5, 1.0])
	plt.yticks([0.1, 0.5, 1.0], [0.1, 0.5, 1.0])
	plt.show()

	""" Plotting loss curve
	"""
	plt.plot(losses)
	plt.show()
	plt.clf()
