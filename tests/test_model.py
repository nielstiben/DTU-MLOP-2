import sys

sys.path.append("./")
import torch
import torch.nn as nn

from vae_mnist import hyperparameter_defaults as config


class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_mean = nn.Linear(hidden_dim, latent_dim)
        self.FC_var = nn.Linear(hidden_dim, latent_dim)
        self.training = True

    def forward(self, x):
        h_ = torch.relu(self.FC_input(x))
        mean = self.FC_mean(h_)
        log_var = self.FC_var(h_)
        std = torch.exp(0.5 * log_var)
        z = self.reparameterization(mean, std)
        return z, mean, log_var

    def reparameterization(
        self,
        mean,
        std,
    ):
        epsilon = torch.rand_like(std)
        z = mean + std * epsilon
        return z


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h = torch.relu(self.FC_hidden(x))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat


def test_model():
    encoder = Encoder(
        input_dim=config.get("x_dim"),
        hidden_dim=config.get("hidden_dim"),
        latent_dim=config.get("latent_dim"),
    )
    decoder = Decoder(
        latent_dim=config.get("latent_dim"),
        hidden_dim=config.get("hidden_dim"),
        output_dim=config.get("x_dim"),
    )
    input = torch.rand(64, 784)
    encoded, _, _ = encoder(input)
    assert encoded.shape == (64, 20)

    decoded = decoder(encoded)
    assert decoded.shape == (64, 784)
