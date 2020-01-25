import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from losses import gaussian_likelihood_sum
import torch.autograd as autograd


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

        g_loss = torch.log(self.epsilon + self.lambda_shared - self.calculate_likelihood(fake)).mean()
        g_loss.backward()
        self.optimizer_g.step()
        return g_loss.item()
