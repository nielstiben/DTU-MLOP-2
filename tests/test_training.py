import sys

import torch
import torch.nn as nn
import torchvision.transforms as transforms

sys.path.append("./")
from pytorch_lightning import LightningModule, Trainer
from test_model import Decoder, Encoder
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from vae_mnist import hyperparameter_defaults as config


def test_training():
    #%% Data loading
    dataset_path = "datasets"
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(dataset_path, transform=mnist_transform, train=False, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.get("batch_size"), shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.get("batch_size"), shuffle=False)

    class Model(LightningModule):
        def __init__(self, Encoder, Decoder):
            super(Model, self).__init__()
            self.Encoder = Encoder
            self.Decoder = Decoder

        def forward(self, x):
            z, mean, log_var = self.Encoder(x)
            x_hat = self.Decoder(z)
            return x_hat, mean, log_var

        def loss_function(self, x, x_hat, mean, log_var):
            reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction="sum")
            KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())
            return reproduction_loss + KLD

        def training_step(self, batch, batch_idx):
            x, _ = batch
            assert x.shape == (100, 1, 28, 28), "Received invalid data dimension"
            x = x.view(config.get("batch_size"), config.get("x_dim"))
            assert x.shape == (100, 784), "Failed to flatten  (100, 1, 28, 28) to (100,784)"
            x = x.to(config.get("device"))
            x_hat, mean, log_var = self(x)
            loss = self.loss_function(x, x_hat, mean, log_var)
            return loss

        def validation_step(self, batch, batch_idx):
            pass

        def configure_optimizers(self):
            return Adam(model.parameters(), lr=config.get("lr"))

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
    model = Model(
        Encoder=encoder,
        Decoder=decoder,
    ).to(config.get("device"))

    trainer = Trainer(max_epochs=1)
    trainer.fit(model, train_loader, test_loader)
