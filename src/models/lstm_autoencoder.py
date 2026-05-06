import numpy as np
import torch
from torch import nn
from tqdm import tqdm


class LSTMAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers):
        super().__init__()
        self.encoder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

    def forward(self, x):
        _, (hidden_state, _) = self.encoder(x)
        latent = self.to_latent(hidden_state[-1])
        decoded_seed = self.from_latent(latent)
        repeated_seed = decoded_seed.unsqueeze(1).repeat(1, x.shape[1], 1)
        reconstructed, _ = self.decoder(repeated_seed)
        return reconstructed


def train_autoencoder(model, train_loader, epochs, lr, device) -> list[float]:
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.to(device)

    for _ in tqdm(range(epochs), desc="Autoencoder epochs"):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            batch_inputs = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(batch_inputs)
            loss = criterion(reconstructed, batch_inputs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch_inputs)

        losses.append(epoch_loss / len(train_loader.dataset))

    return losses


def reconstruction_errors(model, data_loader, device) -> np.ndarray:
    errors = []
    model.eval()

    with torch.no_grad():
        for batch in data_loader:
            batch_inputs = batch[0].to(device)
            reconstructed = model(batch_inputs)
            batch_errors = torch.mean((reconstructed - batch_inputs) ** 2, dim=(1, 2))
            errors.append(batch_errors.cpu().numpy())

    return np.concatenate(errors)
