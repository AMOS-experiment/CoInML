import torch.nn as nn


# Define the DeepAutoencoder class
class DeepAutoencoder(nn.Module):
    def __init__(self, input_dim, latent_dim=7):
        super(DeepAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.SiLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, latent_dim),  # Latent space
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.SiLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 64),
            nn.SiLU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, input_dim),  # Reconstruct original compressed features
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
