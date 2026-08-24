import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import Modelisation.Evaluation.evaluation as ev
from Modelisation.Baselines.baseline import BaselineModel


class TCCMModel:

    def __init__(
        self,
        n_features,
        epochs=100,
        learning_rate=0.001,
        batch_size=64,
        device="cpu",
    ):
        self.device = device
        self.epochs = epochs
        self.lr = learning_rate
        self.batch_size = batch_size
        self.model = FlowMatching_TCCM(
            input_dim=n_features
        ).to(self.device)

    def fit(self, X_train):
        """Train the TCCM model."""
        y_train = torch.zeros(
            X_train.shape[0],
            dtype=torch.long,
            device=self.device,
        ).squeeze()

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
        )

        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
        )
        criterion = nn.MSELoss()

        for _ in range(self.epochs):
            for batch_x, _ in train_loader:
                batch_x = batch_x.to(self.device)

                optimizer.zero_grad()

                t = torch.rand(
                    batch_x.shape[0],
                    1,
                    device=self.device,
                )

                f_xt = self.model(batch_x, t)

                dx_dt = -batch_x
                loss = criterion(f_xt, dx_dt)

                loss.backward()
                optimizer.step()

    def decision_function(self, X_test):
        """Compute anomaly scores for the input samples."""
        with torch.no_grad():
            X_test = X_test.to(self.device)

            t = torch.ones(
                X_test.shape[0],
                1,
                device=self.device,
                dtype=torch.float32,
            )

            f_xt = self.model(X_test, t)
            anomaly_scores = torch.norm(
                f_xt + X_test,
                dim=1,
            )

        return anomaly_scores.cpu().numpy()


class SinusoidalTimeEmbedding(nn.Module):

    def __init__(self, dim, max_period=10000):
        super().__init__()

        self.dim = dim

        half_dim = dim // 2
        freq = torch.exp(
            -math.log(max_period)
            * torch.arange(
                0,
                half_dim,
                dtype=torch.float32,
            )
            / half_dim
        )

        self.register_buffer("freq", freq)

    def forward(self, t):
        t = t.view(-1, 1)
        args = t * self.freq

        return torch.cat(
            [
                torch.sin(args),
                torch.cos(args),
            ],
            dim=-1,
        )


class FlowMatching_TCCM(nn.Module):

    def __init__(self, input_dim, time_embed_dim=128):
        super().__init__()

        self.time_embedding = SinusoidalTimeEmbedding(
            time_embed_dim
        )

        self.model = nn.Sequential(
            nn.Linear(input_dim + time_embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, input_dim),
        )

    def forward(self, x, t):
        t_emb = self.time_embedding(t)
        x_t = torch.cat([x, t_emb], dim=1)

        return self.model(x_t)


class TCCM(BaselineModel):

    def __init__(self, args):
        self.model = TCCMModel(**args)

    def train(self, X_train):
        self.model.fit(X_train)
        return self.model

    def test(self, X_test, y_test):
        scores = self.model.decision_function(X_test)

        auc_tccm, fpr95_tccm, ap_tccm = ev.evaluation(
            y_test,
            scores,
            verbose=False,
        )

        return auc_tccm, fpr95_tccm, ap_tccm