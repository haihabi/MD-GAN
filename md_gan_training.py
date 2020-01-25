import torch
from losses import gaussian_likelihood_sum
import torch.autograd as autograd


def train_critic(input_net_d, input_net_g, input_optimizer_d, real_data, input_batch_size, z_size, input_simplex,
                 epsilon: float,
                 lambda_shared: float,
                 input_working_device: torch.device):
    for p in input_net_d.parameters():  # reset requires_grad
        p.requires_grad = True  # they are set to False below in netG update
    real_data_v = real_data
    input_net_d.zero_grad()
    ######################################################################
    # train with real
    ######################################################################
    e_real = input_net_d(real_data_v)
    lk_real = gaussian_likelihood_sum(e_real, input_simplex)
    d_real_loss = -torch.log(epsilon + lk_real).mean()
    d_real_loss.backward()
    ######################################################################
    # train with fake
    ######################################################################
    noise = torch.randn(input_batch_size, z_size).to(input_working_device)
    with torch.no_grad():
        noisev = autograd.Variable(noise).to(input_working_device)  # totally freeze netG
    fake = autograd.Variable(input_net_g(noisev).data).to(input_working_device)

    e_fake = input_net_d(fake)
    lk_fake = gaussian_likelihood_sum(e_fake, input_simplex)
    d_fake_loss = -torch.log(epsilon + lambda_shared - lk_fake).mean()
    d_fake_loss.backward()
    ######################################################################
    # calculate loss function and update weights
    ######################################################################
    d_cost = (d_fake_loss + d_real_loss).item()
    input_optimizer_d.step()
    return d_cost


def train_generator(input_net_d, input_net_g, input_optimizer_g, input_batch_size, z_size, input_simplex,
                    epsilon: float,
                    lambda_shared: float,
                    input_working_device):
    for p in input_net_d.parameters():
        p.requires_grad = False  # to avoid computation
    input_net_g.zero_grad()

    noise = autograd.Variable(torch.randn(input_batch_size, z_size)).to(input_working_device)
    fake = input_net_g(noise)

    e_generator = input_net_d(fake)
    lk_fake = gaussian_likelihood_sum(e_generator, input_simplex)
    g_loss = torch.log(epsilon + lambda_shared - lk_fake).mean()
    g_loss.backward()
    input_optimizer_g.step()
    return g_loss.item()
