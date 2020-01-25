import torch
from torch import optim
from simplex_generator import simplex_params
from losses import gaussian_likelihood_sum
from architectures import LambdaNetwork, Discriminator, Generator
from gmm_data_generator import load_db
from numpy_dataset import NumpyDataset
from md_gan_training import train_critic, train_generator

PARAMS = {'batch_size': 500,
          'zdim': 2,
          'eta_lambda': 0.01,
          'e_dim': 9,
          'lr_d': 1e-3,
          'lr_g': 1e-3,
          'epsilon': 1e-8,  # for avoiding numerical instabilities
          'samp_num_gen': 2500}
# TODO: move all numbers to params dict
###############################
# Get working device (cpu/gpu)
###############################
working_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Current Working Device is set to:" + str(working_device))
###############################
# Create simplex
###############################
simplex = simplex_params(PARAMS['e_dim'], working_device)
###############################
# Train Lambda Network
# TODO: move to function
###############################
lambda_net = LambdaNetwork(PARAMS['e_dim']).to(working_device)
lambda_training_data = torch.tensor([1.0], device=working_device, dtype=torch.float32, requires_grad=False)
optimizer_lambda = optim.Adam(lambda_net.parameters(), lr=PARAMS['eta_lambda'])

for i in range(10001):
    optimizer_lambda.zero_grad()
    e = lambda_net(lambda_training_data)
    lambda_lk = gaussian_likelihood_sum(e, simplex)
    lambda_loss = -torch.log(PARAMS['epsilon'] + lambda_lk).mean()
    if i % 1000 == 0 and i > 999:
        print("Lambda Loss:" + str(lambda_loss.item()))
        for group in optimizer_lambda.param_groups:
            group['lr'] = group['lr'] * 0.5
    lambda_loss.backward()
    optimizer_lambda.step()
e = lambda_net(lambda_training_data)
lambda_value = gaussian_likelihood_sum(e, simplex).sum().item()
print(lambda_value)
####################################
# Train Generator and Discriminator
####################################
net_g = Generator(PARAMS['zdim']).to(working_device)
net_d = Discriminator(PARAMS['e_dim']).to(working_device)
optimizer_g = optim.Adam(net_g.parameters(), lr=PARAMS['lr_g'], betas=(0.5, 0.999))
optimizer_d = optim.Adam(net_d.parameters(), lr=PARAMS['lr_d'], betas=(0.5, 0.999))

lr_g = optim.lr_scheduler.MultiStepLR(optimizer_g, [500, 1000, 1500], gamma=0.5)
lr_d = optim.lr_scheduler.MultiStepLR(optimizer_d, [500, 1000, 1500], gamma=0.5)

training_data = load_db()
train_dataset = NumpyDataset(training_data)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=PARAMS['batch_size'],
                                           shuffle=True)
# Start Trainnig Loop
for epoch in range(2000):  # TODO:move to param
    for d in train_loader:
        d = d.to(working_device)
        train_critic(net_d, net_g, optimizer_d, d, PARAMS['batch_size'], PARAMS['zdim'], simplex,
                     PARAMS['epsilon'],
                     lambda_value,
                     working_device)
        train_generator(net_d, net_g, optimizer_g, PARAMS['batch_size'], PARAMS['zdim'], simplex,
                        PARAMS['epsilon'],
                        lambda_value,
                        working_device)
    lr_g.step(epoch)
    lr_d.step(epoch)
######################################
# Plot
######################################
from matplotlib import pyplot as plt

g_samples = net_g(torch.randn(PARAMS['batch_size'], PARAMS['zdim']).to(working_device))
g_samples = g_samples.cpu().detach().numpy()

plt.plot(training_data[:, 0], training_data[:, 1], 'o')
plt.plot(g_samples[:, 0], g_samples[:, 1], '^')
plt.show()
