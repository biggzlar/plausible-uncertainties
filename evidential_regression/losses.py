import torch


def NIG_NLL(y_true, gamma, nu, alpha, beta, reduce=False):
    """ The univariate formulation of deep evidential regression, including the regularization
        term, is taken directly from https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html
        as is the accompanying code: https://github.com/aamini/evidential-deep-learning.

        We replace the final line of the loss function with Stirling's approximation for
        simplicity and smoothness.
    """
    two_b_lambda = 1e-8 + 2 * beta * (1 + nu)
    nll = 0.5 * torch.log(torch.pi / (nu + 1e-8)) \
        + 0.5 * torch.log(two_b_lambda) \
        + (alpha + 0.5) * torch.log(1 + (nu * torch.square(y_true - gamma)) / two_b_lambda) \
        - 0.5 * torch.log(alpha)
    return torch.mean(nll) if reduce else nll


def NIG_REG(y_true, gamma, nu, alpha, beta, reduce=False):
    """ The regularizer as proposed by Amini et al. is adapted to maximize
        calibration and accurate recovery of the aleatoric component. Excluding 
        $\nu$ from the total evidence term improves aleatoric uncertainty estimates.

        Dividing the error term by the aleatoric component may increase 
        calibration scores in some instances.
    """
    error = torch.abs(y_true - gamma) / (beta * torch.reciprocal(alpha - 1.0))
    evi = 2 * alpha
    reg = error * evi
    
    return torch.mean(reg) if reduce else reg


class UnivariateEvidentialRegressionLoss(torch.nn.Module):
    def __init__(self):
        super(UnivariateEvidentialRegressionLoss, self).__init__()

    def forward(self, y_true, gamma, nu, alpha, beta, mask=None, coeff=1e-2):        
        if mask is not None:
            y_true = y_true[mask]
            gamma = gamma[mask]
            nu = nu[mask]
            alpha = alpha[mask]
            beta = beta[mask]

        loss_nll = NIG_NLL(y_true, gamma, nu, alpha, beta)
        loss_reg = NIG_REG(y_true, gamma, nu, alpha, beta)
        loss = torch.mean(loss_nll + coeff * loss_reg)
        return loss


def NIW_NLL(y_true, mu, nu, kappa, L, p):
    residuals = y_true - mu
    pairwise_deviation_products = torch.einsum('bj, bi -> bji', residuals, residuals)
    sigma = torch.einsum('bij, bjk -> bik', L, torch.transpose(L, -2, -1))
    
    nll = torch.lgamma((nu - p + 1) / 2) - torch.lgamma((nu + 1) / 2) \
        + (p / 2) * torch.log(((kappa + 1) / kappa) * torch.pi) \
        - nu * torch.log(L.diagonal(offset=0, dim1=-1, dim2=-2)).sum(-1).unsqueeze(dim=-1) \
        + ((nu + 1) / 2) * torch.logdet(sigma + (pairwise_deviation_products * kappa.unsqueeze(-1) / (1 + kappa.unsqueeze(-1)))).unsqueeze(-1)

    return nll


def NIW_REG(y_true, mu, nu, kappa):
    error = torch.abs(y_true - mu)
    evi = nu + kappa

    return evi * error


class MultivariateEvidentialRegressionLoss(torch.nn.Module):
    def __init__(self, p=2):
        super(MultivariateEvidentialRegressionLoss, self).__init__()
        self.p = p

    def forward(self, y_true, mu, nu, kappa, L, mask=None, coeff=0.0):        
        if mask is not None:
            y_true = y_true[mask]
            mu = mu[mask]
            nu = nu[mask]
            kappa = kappa[mask]
            L = L[mask]

        loss_nll = NIW_NLL(y_true, mu, nu, kappa, L, self.p)
        # loss_reg = NIW_REG(y_true, mu, nu, kappa)
        
        loss = torch.mean(loss_nll)
        return loss
