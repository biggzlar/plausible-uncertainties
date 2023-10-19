import torch
import torch.nn as nn


class UnivariateL1Loss(nn.Module):
    def __init__(self):
        super(UnivariateL1Loss, self).__init__()
    
    def forward(self, y_true, y_pred, sigma):
        return torch.mean((0.5 * torch.exp(-sigma)) * torch.abs(y_pred - y_true) + 0.5 * sigma)


class UnivariateL2Loss(nn.Module):
    def __init__(self):
        super(UnivariateL2Loss, self).__init__()
    
    def forward(self, y_pred, y_true, sigma):
        return torch.mean((0.5 * torch.exp(-sigma)) * torch.norm(y_pred - y_true, p=2, dim=-1) + 0.5 * sigma)


class BetaNLLLoss(nn.Module):
    """ Based on https://arxiv.org/abs/2203.09168 by Seitzer et al.
    """
    def __init__(self):
        super(BetaNLLLoss, self).__init__()
    
    def forward(self, y_pred, y_true, sigma, beta=0.5):
        return torch.mean(torch.exp(sigma.detach())**(2 * beta) * ((0.5 * torch.exp(-sigma)) * torch.norm(y_pred - y_true, p=2, dim=-1) + 0.5 * sigma))


class MultivariateGaussianNLL(nn.Module):
    def __init__(self):
        super(MultivariateGaussianNLL, self).__init__()

    def forward(self, y_pred, y_true, L):
        residuals = y_pred - y_true
        precision = torch.einsum('bij, bjk -> bik', L, torch.transpose(L, -2, -1))

        weighted_residuals = torch.einsum('bij,bj -> bi', precision, residuals)
        sample_loss = torch.einsum('bi,bi -> b', residuals, weighted_residuals)
        loss = sample_loss - torch.log(torch.det(precision))

        return loss.mean()