import torch
import torch.nn as nn
import torchvision.transforms as transforms
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

import wandb

hyperparameter_defaults = dict(
    batch_size=100,
    epochs=5,
    dataset_path="datasets",
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    x_dim=784,
    hidden_dim=400,
    latent_dim=20,
    lr=1e-3,
)


def main():
    wandb.init(config=hyperparameter_defaults)
    config = wandb.config

    #%% Data loading
    mnist_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = MNIST(config.dataset_path, transform=mnist_transform, train=True, download=True)
    test_dataset = MNIST(config.dataset_path, transform=mnist_transform, train=False, download=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=config.batch_size, shuffle=False)

    #%% Models
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, latent_dim):
            super(Encoder, self).__init__()
            self.FC_input = nn.Linear(input_dim, hidden_dim)
            self.FC_mean = nn.Linear(hidden_dim, latent_dim)
            self.FC_var = nn.Linear(hidden_dim, latent_dim)
            self.training = True

        def forward(self, x):
            if x.ndim != 2:
                raise ValueError("Expected input to be a 2d tensor")
            if x.shape[1] != 784:
                raise ValueError("Expected input shape is [batch_size, 784]")
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
            if x.ndim != 2:
                raise ValueError("Expected input to be a 2d tensor")
            if x.shape[1] != 20:
                raise ValueError("Expected input shape is [batch_size, 20]")
            h = torch.relu(self.FC_hidden(x))
            x_hat = torch.sigmoid(self.FC_output(h))
            return x_hat

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
            x = x.view(config.batch_size, config.x_dim)
            x = x.to(config.device)
            x_hat, mean, log_var = self(x)
            loss = self.loss_function(x, x_hat, mean, log_var)
            wandb.log({"Loss": loss})
            return loss

        def validation_step(self, batch, batch_idx):
            x, _ = batch
            with torch.no_grad():
                for batch_idx, (x, _) in enumerate(test_loader):
                    x = x.view(config.batch_size, config.x_dim)
                    x = x.to(config.device)
                    x_encoded, _, _ = encoder(x)
                    x_encoded_decoded = decoder(x_encoded)  # Same as x_hat
                    noise = torch.randn(config.batch_size, config.latent_dim).to(config.device)
                    x_noise_decoded = decoder(noise)
                    break
            # Original
            wandb.log({"Original Images": wandb.Image(x.view(config.batch_size, 1, 28, 28))})
            # Encoded -> Decoded (x_hat)
            wandb.log(
                {
                    "Reconstructed Images (first Encoded, then Decoded)": wandb.Image(
                        x_encoded_decoded.view(config.batch_size, 1, 28, 28)
                    )
                }
            )
            # Decoded from random noice
            wandb.log(
                {
                    "Generated Images (random noise trough Decoder)": wandb.Image(
                        x_noise_decoded.view(config.batch_size, 1, 28, 28)
                    )
                }
            )

        def configure_optimizers(self):
            return Adam(model.parameters(), lr=config.lr)

    encoder = Encoder(
        input_dim=config.x_dim,
        hidden_dim=config.hidden_dim,
        latent_dim=config.latent_dim,
    )
    decoder = Decoder(
        latent_dim=config.latent_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.x_dim,
    )
    model = Model(
        Encoder=encoder,
        Decoder=decoder,
    ).to(config.device)
    wandb.watch(model, log_freq=100)

    #%% Trainer
    checkpoint_callback = ModelCheckpoint(dirpath="./models", monitor="val_loss", mode="min")
    trainer = Trainer(callbacks=[checkpoint_callback], max_epochs=config.epochs)
    trainer.fit(model, train_loader, test_loader)


if __name__ == "__main__":
    main()
