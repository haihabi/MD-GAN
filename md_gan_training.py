import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from losses import gaussian_likelihood_sum
import torch.autograd as autograd


#
# def enable_gradients(input_module):
#     for p in input_module.parameters():  # reset requires_grad
#         p.requires_grad = True  # they are set to False below in netG update
#
#
# def disable_gradients(input_module):
#     for p in input_module.parameters():  # reset requires_grad
#         p.requires_grad = False  # they are set to False below in netG update


class MDGANTraining(object):
    def __init__(self, input_net_d: nn.Module, input_net_g: nn.Module, input_optimizer_d: Optimizer,
                 input_optimizer_g: Optimizer,
                 input_batch_size: int, z_size: int,
                 input_simplex,
                 epsilon: float,
                 lambda_shared: float,
                 input_working_device: torch.device):
        self.working_device = input_working_device
        self.lambda_shared = lambda_shared
        self.epsilon = epsilon
        self.simplex = input_simplex
        self.batch_size = input_batch_size
        self.z_size = z_size
        self.optimizer_d = input_optimizer_d
        self.optimizer_g = input_optimizer_g
        self.net_g = input_net_g
        self.net_d = input_net_d

    def enable_gradients(self):
        for p in self.net_d.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

    def disable_gradients(self):
        for p in self.net_d.parameters():  # reset requires_grad
            p.requires_grad = False  # they are set to False below in netG update

    def samples_noise(self):
        return torch.randn(self.batch_size, self.z_size).to(self.working_device)

    def calculate_likelihood(self, data: torch.Tensor):
        e = self.net_d(data)
        return gaussian_likelihood_sum(e, self.simplex)

    def update_discriminator(self, real_data: torch.Tensor):
        self.enable_gradients()
        self.net_d.zero_grad()
        ######################################################################
        # train with real
        ######################################################################
        d_real_loss = -torch.log(self.epsilon + self.calculate_likelihood(real_data)).mean()
        d_real_loss.backward()
        ######################################################################
        # train with fake
        ######################################################################
        noise = self.samples_noise()
        with torch.no_grad():
            noisev = autograd.Variable(noise).to(self.working_device)  # totally freeze netG
        fake = autograd.Variable(self.net_g(noisev).data).to(self.working_device)

        d_fake_loss = -torch.log(self.epsilon + self.lambda_shared - self.calculate_likelihood(fake)).mean()
        d_fake_loss.backward()
        ######################################################################
        # calculate loss function and update weights
        ######################################################################
        self.optimizer_d.step()
        return (d_fake_loss + d_real_loss).item()

    def update_generator(self):
        self.disable_gradients()

        self.net_g.zero_grad()
        noise = self.samples_noise()

        fake = self.net_g(noise)

        # e_generator = self.net_d(fake)
        # lk_fake = gaussian_likelihood_sum(e_generator, self.simplex)
        g_loss = torch.log(self.epsilon + self.lambda_shared - self.calculate_likelihood(fake)).mean()
        g_loss.backward()
        self.optimizer_g.step()
        return g_loss.item()

# def train_discriminator(input_net_d: nn.Module, input_net_g: nn.Module, input_optimizer_d, real_data: torch.Tensor,
#                         input_batch_size: int, z_size: int,
#                         input_simplex,
#                         epsilon: float,
#                         lambda_shared: float,
#                         input_working_device: torch.device):
#     # for p in input_net_d.parameters():  # reset requires_grad
#     #     p.requires_grad = True  # they are set to False below in netG update
#     enable_gradients(input_net_d)
#     # real_data_v = real_data
#     input_net_d.zero_grad()
#     ######################################################################
#     # train with real
#     ######################################################################
#     e_real = input_net_d(real_data)
#     lk_real = gaussian_likelihood_sum(e_real, input_simplex)
#     d_real_loss = -torch.log(epsilon + lk_real).mean()
#     d_real_loss.backward()
#     ######################################################################
#     # train with fake
#     ######################################################################
#     noise = torch.randn(input_batch_size, z_size).to(input_working_device)
#     with torch.no_grad():
#         noisev = autograd.Variable(noise).to(input_working_device)  # totally freeze netG
#     fake = autograd.Variable(input_net_g(noisev).data).to(input_working_device)
#
#     e_fake = input_net_d(fake)
#     lk_fake = gaussian_likelihood_sum(e_fake, input_simplex)
#     d_fake_loss = -torch.log(epsilon + lambda_shared - lk_fake).mean()
#     d_fake_loss.backward()
#     ######################################################################
#     # calculate loss function and update weights
#     ######################################################################
#     d_cost = (d_fake_loss + d_real_loss).item()
#     input_optimizer_d.step()
#     return d_cost
#
#
# def train_generator(input_net_d, input_net_g, input_optimizer_g, input_batch_size, z_size, input_simplex,
#                     epsilon: float,
#                     lambda_shared: float,
#                     input_working_device):
#     disable_gradients(input_net_d)
#
#     input_net_g.zero_grad()
#     noise = autograd.Variable(torch.randn(input_batch_size, z_size)).to(input_working_device)
#     fake = input_net_g(noise)
#
#     e_generator = input_net_d(fake)
#     lk_fake = gaussian_likelihood_sum(e_generator, input_simplex)
#     g_loss = torch.log(epsilon + lambda_shared - lk_fake).mean()
#     g_loss.backward()
#     input_optimizer_g.step()
#     return g_loss.item()
