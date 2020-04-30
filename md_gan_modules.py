import torch
from torch import nn


class ScaledSigmoid(nn.Module):
    def __init__(self, scale: float = 5.0, shift: float = 2.5):
        """
        The Scaled Sigmoid Module
        :param scale: a float scale value
        :param shift: a float shift value
        """
        super(ScaledSigmoid, self).__init__()
        self.scale = scale
        self.shift = shift

    def forward(self, x):
        return self.scale * torch.sigmoid(x) - self.shift


class ScaledTanh(nn.Module):
    def __init__(self, scale: float = 6.0):
        """
        The Scaled Sigmoid Module
        :param scale: a float scale value
        """
        super(ScaledTanh, self).__init__()
        self.scale = scale

    def forward(self, x):
        return self.scale * torch.tanh(x)


class FeedForwardSequential(nn.Module):
    def __init__(self, input_dim: int, layers_dim: list, non_linear: nn.Module, output_non_linear: nn.Module):
        super(FeedForwardSequential, self).__init__()
        layers_dim = [input_dim, *layers_dim]
        layers_list = []
        for i in range(len(layers_dim) - 1):
            layers_list.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            layers_list.append(output_non_linear() if i == (len(layers_dim) - 2) else non_linear())
        self.layer_seq = nn.Sequential(*layers_list)
        self.initialization()

    def initialization(self):
        for layer in self.layer_seq:
            if isinstance(layer, nn.Linear):
                torch.nn.init.normal_(layer.weight, mean=0, std=0.02)

    def forward(self, x):
        return self.layer_seq(x)


class Generator(nn.Module):
    def __init__(self, z_dim, layers_dim: list = (128, 128, 2), non_linear=nn.ReLU, output_non_linear=ScaledTanh):
        super(Generator, self).__init__()
        self.feed_forward = FeedForwardSequential(input_dim=z_dim, layers_dim=layers_dim, non_linear=non_linear,
                                                  output_non_linear=output_non_linear)

    def forward(self, x):
        return self.feed_forward(x)


def leaky_relu():
    return nn.LeakyReLU(negative_slope=0.2)


class Discriminator(nn.Module):
    def __init__(self, e_dim, layers_dim: list = (128, 128), non_linear=leaky_relu,
                 output_non_linear=ScaledSigmoid, x_dim=2):
        super(Discriminator, self).__init__()
        layers_dim = [*list(layers_dim), e_dim]
        self.feed_forward = FeedForwardSequential(input_dim=x_dim, layers_dim=layers_dim, non_linear=non_linear,
                                                  output_non_linear=output_non_linear)

    def forward(self, x):
        return self.feed_forward(x)


class LambdaNetwork(nn.Module):
    def __init__(self, gmm_dim):
        super(LambdaNetwork, self).__init__()
        self.feed_forward = nn.Sequential(nn.Linear(1, gmm_dim), ScaledSigmoid())

    def forward(self, x):
        x = x.reshape(-1, 1)
        return self.feed_forward(x)
