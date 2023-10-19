import torch


class DenseInverseGamma(torch.nn.Module):
    """ Based on: https://github.com/aamini/evidential-deep-learning.
    """
    def __init__(self, in_features, units=1):
        super(DenseInverseGamma, self).__init__()
        self.units = units
        self.dense = torch.nn.Linear(in_features=in_features, out_features=4 * self.units)
        self.softplus = torch.nn.Softplus()

    def evidence(self, x):
        return self.softplus(x)

    def forward(self, x):
        output = self.dense(x)
        mu, logv, logalpha, logbeta = torch.split(output, split_size_or_sections=self.units, dim=-1)
        
        nu = self.evidence(logv)
        alpha = self.evidence(logalpha) + 2
        beta = self.evidence(logbeta)
        
        return mu, nu, alpha, beta


class DenseInverseWishart(torch.nn.Module):
    def __init__(self, in_features, p=1, mu_activation=None):
        super(DenseInverseWishart, self).__init__()
        self.p = p
        self.diag_indices = [i for i in range(self.p)]
        self.tril_indices = torch.tril_indices(self.p, self.p).tolist()

        self.mu = torch.nn.Linear(in_features=in_features, out_features=self.p)
        self.params = torch.nn.Linear(in_features=in_features, out_features=2)

        self.n_decomposit_units = int((1 + self.p) * self.p / 2)
        self.L_decomposit = torch.nn.Linear(in_features=in_features, out_features=self.p**2)
        
        self.softplus = torch.nn.Softplus()
        self.mu_activation = mu_activation

    def evidence(self, x):
        return self.softplus(x)

    def forward(self, x):
        mu = self.mu(x)
        params = self.params(x)
        lognu, logkappa = torch.split(params, split_size_or_sections=1, dim=-1)
        
        if self.mu_activation is not None:
            mu = self.mu_activation(mu)
        nu = self.evidence(lognu) + self.p + 1
        kappa = self.evidence(logkappa) + 1
        
        L = self.L_decomposit(x)
        L = L.view(-1, self.p, self.p)
        L = torch.tril(L, diagonal=-1) + torch.diag_embed(1e-2 + self.evidence(torch.diagonal(L, dim1=-2, dim2=-1)))

        # non_zeros = self.L_decomposit(x)
        # L = torch.zeros((x.shape[0], self.p, self.p))
        # L[:, self.tril_indices[0], self.tril_indices[1]] = non_zeros
        # L[:, self.diag_indices, self.diag_indices] = self.evidence(L[:, self.diag_indices, self.diag_indices])

        return mu, nu, kappa, L