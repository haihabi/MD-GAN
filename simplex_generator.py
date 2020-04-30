import numpy as np
import torch
from dataclasses import dataclass


def simplex_coordinates(m):
    # This function is adopted from the Simplex Coordinates library
    # https://people.sc.fsu.edu/~jburkardt/py_src/simplex_coordinates/simplex_coordinates.html
    x = np.zeros([m, m + 1])

    for j in range(0, m):
        x[j, j] = 1.0

    a = (1.0 - np.sqrt(float(1 + m))) / float(m)

    for i in range(0, m):
        x[i, m] = a
    c = np.zeros(m)
    for i in range(0, m):
        s = 0.0
        for j in range(0, m + 1):
            s = s + x[i, j]
        c[i] = s / float(m + 1)

    for j in range(0, m + 1):
        for i in range(0, m):
            x[i, j] = x[i, j] - c[i]
    s = 0.0
    for i in range(0, m):
        s = s + x[i, 0] ** 2
        s = np.sqrt(s)

    for j in range(0, m + 1):
        for i in range(0, m):
            x[i, j] = x[i, j] / s
    return x


def var2cov(bot_dim, ngmm):
    cov = np.zeros((bot_dim, bot_dim))
    for k_ in range(bot_dim):
        cov[k_, k_] = 1.
    sigma_real_batch = []
    for c in range(ngmm):
        sigma_real_batch.append(cov)
    return np.array(sigma_real_batch, dtype=np.float32).squeeze().astype('float32') * .25


@dataclass
class Simplex:
    mu: torch.Tensor
    sigma: torch.Tensor
    w: torch.Tensor
    sigma_det_rsqrt: torch.Tensor
    sigma_inv: torch.Tensor


def simplex_params(bot_dim: int, input_working_device: torch.device) -> Simplex:
    ngmm = bot_dim + 1
    mu_real_batch = simplex_coordinates(bot_dim)
    sigma_real = var2cov(bot_dim, ngmm).astype('float32')
    mu_real = np.array(mu_real_batch.T, dtype=np.float32)
    w_real = (np.ones((ngmm,)) / ngmm).astype('float32')
    sigma_det_rsqrt = np.power(2 * np.pi * np.linalg.det(sigma_real), -0.5)
    sigma_inv = np.linalg.inv(sigma_real)
    ##########################################
    # Change to torch tensor
    ##########################################
    mu_simplex = torch.tensor(mu_real, device=input_working_device, dtype=torch.float32)
    sigma_simplex = torch.tensor(sigma_real, device=input_working_device, dtype=torch.float32)
    w_simplex = torch.tensor(w_real, device=input_working_device, dtype=torch.float32)
    sigma_det_rsqrt = torch.tensor(sigma_det_rsqrt, device=input_working_device, dtype=torch.float32)
    sigma_inv = torch.tensor(sigma_inv, device=input_working_device, dtype=torch.float32)
    return Simplex(mu_simplex, sigma_simplex, w_simplex, sigma_det_rsqrt, sigma_inv)
