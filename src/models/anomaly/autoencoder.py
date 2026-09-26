"""Deep Autoencoder for anomaly detection using PyTorch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam

from src.models.base import BaseModelWrapper


class Autoencoder(nn.Module):
    """Multi-layer autoencoder with dropout."""

    def __init__(self, input_dim: int, encoding_dim: int = 16, dropout: float = 0.2) -> None:
        super().__init__()
        hidden_dims = [64, 32]
        bottleneck_dim = encoding_dim

        encoder_layers: list[nn.Module] = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        encoder_layers.append(nn.Linear(prev_dim, bottleneck_dim))

        decoder_layers: list[nn.Module] = []
        prev_dim = bottleneck_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))

        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DeepAutoencoderModel(BaseModelWrapper):
    """Deep autoencoder for unsupervised anomaly detection.

    Trained on a healthy (normal) cohort to learn the normal
    physiological distribution. Reconstruction MSE is used as
    the anomaly score — higher MSE means more anomalous.

    Args:
        encoding_dim: Size of the bottleneck layer.
        dropout: Dropout rate for regularization.
        patience: Early stopping patience.
        learning_rate: Adam learning rate.
        max_epochs: Maximum training epochs.
        batch_size: Training batch size.
    """

    def __init__(
        self,
        input_dim: int | None = None,
        encoding_dim: int = 16,
        dropout: float = 0.2,
        patience: int = 5,
        learning_rate: float = 0.001,
        max_epochs: int = 100,
        batch_size: int = 256,
    ) -> None:
        params = {
            "encoding_dim": encoding_dim,
            "dropout": dropout,
            "patience": patience,
            "learning_rate": learning_rate,
            "max_epochs": max_epochs,
            "batch_size": batch_size,
        }
        super().__init__(name="deep_autoencoder", params=params)
        self._input_dim = input_dim

    def fit(self, X: pd.DataFrame | np.ndarray, y: np.ndarray | None = None) -> DeepAutoencoderModel:
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        if self._input_dim is None:
            self._input_dim = X_values.shape[1]

        self.model = Autoencoder(
            input_dim=self._input_dim,
            encoding_dim=self.params["encoding_dim"],
            dropout=self.params["dropout"],
        )

        device = torch.device("cpu")
        self.model.to(device)

        X_tensor = torch.FloatTensor(X_values).to(device)

        optimizer = Adam(
            self.model.parameters(),
            lr=self.params["learning_rate"],
        )
        criterion = nn.MSELoss()

        batch_size = self.params["batch_size"]
        max_epochs = self.params["max_epochs"]
        patience = self.params["patience"]

        best_loss = float("inf")
        patience_counter = 0
        self.model.train()
        best_state: dict[str, torch.Tensor] | None = None

        for epoch in range(max_epochs):
            permutation = torch.randperm(X_tensor.size(0))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, X_tensor.size(0), batch_size):
                indices = permutation[i:i + batch_size]
                batch = X_tensor[indices]

                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        self._is_fitted = True
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Return reconstructed values."""
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        self.model.eval()
        device = torch.device("cpu")
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        X_tensor = torch.FloatTensor(X_values).to(device)
        with torch.no_grad():
            reconstructed = self.model(X_tensor)
        return reconstructed.cpu().numpy()

    def get_anomaly_score(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Compute reconstruction MSE as anomaly score.

        Higher MSE means more anomalous.
        """
        if not self._is_fitted:
            raise RuntimeError("Model not fitted yet")
        X_values = X.values if isinstance(X, pd.DataFrame) else X
        reconstructed = self.predict(X_values)
        mse = np.mean((reconstructed - X_values) ** 2, axis=1)
        return mse

    def get_reconstruction_error(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self.get_anomaly_score(X)
