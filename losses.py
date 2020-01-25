import torch


def gaussian_likelihood_sum(e, simplex):
    e_center = (e.unsqueeze(dim=1) - simplex.mu.unsqueeze(dim=0)).unsqueeze(dim=-1)
    exp_value = torch.exp(-0.5 * torch.matmul(torch.matmul(e_center.transpose(-1, -2), simplex.sigma_inv), e_center))
    sigma_det_rsqrt = simplex.sigma_det_rsqrt.reshape(1, -1, 1, 1)
    w = simplex.w.reshape(1, -1, 1, 1)
    likelihood = (w * sigma_det_rsqrt * exp_value).sum(dim=1).reshape(-1)
    return likelihood
