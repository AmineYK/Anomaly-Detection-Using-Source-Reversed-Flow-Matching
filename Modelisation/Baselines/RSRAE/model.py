import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import Modelisation.Evaluation.evaluation as ev
from Modelisation.Baselines.baseline import BaselineModel


class CAE(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_layer_sizes,
        intrinsic_size,
        norm_type="MSE",
        loss_norm_type="MSE",
        activation=F.relu,
        if_rsr=True,
        enforce_proj=False,
        all_alt=False,
        learning_rate=1e-3,
        lambda1=0.1,
        lambda2=0.1,
        epoch_size=50,
        batch_show=25,
        normalize=False,
        bn=True,
        seed=123,
    ):
        super().__init__()

        self.input_dim = int(input_dim)
        self.hidden_layer_sizes = [int(h) for h in hidden_layer_sizes]
        self.intrinsic_size = int(intrinsic_size)

        self.activation = activation
        self.norm_type = norm_type
        self.loss_norm_type = loss_norm_type
        self.if_rsr = if_rsr
        self.enforce_proj = enforce_proj
        self.all_alt = all_alt
        self.learning_rate = learning_rate
        self.epoch_size = epoch_size
        self.batch_show = batch_show
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.normalize = normalize
        self.bn = bn

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.encoder_fc1 = nn.Linear(
            self.input_dim,
            self.hidden_layer_sizes[0],
        )
        self.encoder_fc2 = nn.Linear(
            self.hidden_layer_sizes[0],
            self.hidden_layer_sizes[1],
        )
        self.encoder_fc3 = nn.Linear(
            self.hidden_layer_sizes[1],
            self.hidden_layer_sizes[2],
        )

        if bn:
            self.bn1 = nn.BatchNorm1d(self.hidden_layer_sizes[0])
            self.bn2 = nn.BatchNorm1d(self.hidden_layer_sizes[1])
            self.bn3 = nn.BatchNorm1d(self.hidden_layer_sizes[2])

        self.A = nn.Parameter(
            torch.randn(
                self.hidden_layer_sizes[2],
                self.intrinsic_size,
            )
        )

        self.decoder_fc3 = nn.Linear(
            self.intrinsic_size,
            self.hidden_layer_sizes[2],
        )
        self.decoder_fc2 = nn.Linear(
            self.hidden_layer_sizes[2],
            self.hidden_layer_sizes[1],
        )
        self.decoder_fc1 = nn.Linear(
            self.hidden_layer_sizes[1],
            self.hidden_layer_sizes[0],
        )
        self.decoder_fc0 = nn.Linear(
            self.hidden_layer_sizes[0],
            self.input_dim,
        )

        if bn:
            self.dbn1 = nn.BatchNorm1d(self.hidden_layer_sizes[2])
            self.dbn2 = nn.BatchNorm1d(self.hidden_layer_sizes[1])
            self.dbn3 = nn.BatchNorm1d(self.hidden_layer_sizes[0])

        self.loss_fn = nn.MSELoss()

    def encoder(self, x):
        z = self.activation(self.encoder_fc1(x))

        if self.bn:
            z = self.bn1(z)

        z = self.activation(self.encoder_fc2(z))

        if self.bn:
            z = self.bn2(z)

        z = self.activation(self.encoder_fc3(z))

        if self.bn:
            z = self.bn3(z)

        return z

    def rsr(self, y):
        return y @ self.A

    def renormalization(self, z):
        return F.normalize(z, p=2, dim=-1)

    def decoder(self, z):
        z = self.activation(self.decoder_fc3(z))

        if self.bn:
            z = self.dbn1(z)

        z = self.activation(self.decoder_fc2(z))

        if self.bn:
            z = self.dbn2(z)

        z = self.activation(self.decoder_fc1(z))
        x_hat = self.decoder_fc0(z)

        return x_hat

    def forward(self, x):
        y = self.encoder(x)
        y_rsr = self.rsr(y)

        if self.normalize:
            z = self.renormalization(y_rsr)
        else:
            z = y_rsr

        x_hat = self.decoder(z)

        return y, y_rsr, z, x_hat

    def reconstruction_error(self, x, x_hat):
        if self.loss_norm_type.lower() in ["mse", "f", "frob"]:
            return self.loss_fn(x, x_hat)

        if self.loss_norm_type.lower() == "l1":
            return torch.mean(torch.norm(x - x_hat, p=1, dim=1))

        return torch.mean(torch.norm(x - x_hat, dim=1))

    def pca_error(self, y, z):
        z_proj = z @ self.A.T
        z_proj = y @ (self.A @ self.A.T)

        if self.norm_type.lower() in ["mse", "f", "frob"]:
            return self.loss_fn(y, z_proj)

        if self.norm_type.lower() == "l1":
            return torch.mean(torch.norm(y - z_proj, p=1, dim=1))

        return torch.mean(torch.norm(y - z_proj, dim=1))

    def proj_error(self):
        I = torch.eye(self.A.size(1), device=self.A.device)
        return torch.mean((self.A.T @ self.A - I) ** 2)

    def fit(self, X, batch_size=128, x_val=None, device="cuda"):
        self.to(device)

        n_samples = X.shape[0]
        n_batch = (n_samples - 1) // batch_size + 1

        optimizer_main = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )
        optimizer_proj = torch.optim.Adam(
            [self.A],
            lr=self.learning_rate * 10,
        )

        for epoch in range(self.epoch_size):
            idx = np.random.permutation(n_samples)

            for batch_idx in range(n_batch):
                i_start = batch_idx * batch_size
                i_end = min(
                    (batch_idx + 1) * batch_size,
                    n_samples,
                )

                x_batch = X[idx[i_start:i_end]].to(device)

                optimizer_main.zero_grad()

                y_flat, y_rsr, z, x_hat = self.forward(x_batch)

                loss = self.reconstruction_error(x_batch, x_hat)

                if self.if_rsr and not self.all_alt:
                    loss += (
                        self.lambda1 * self.pca_error(y_flat, y_rsr)
                        + self.lambda2 * self.proj_error()
                    )

                loss.backward()
                optimizer_main.step()

                if self.enforce_proj and self.all_alt:
                    optimizer_proj.zero_grad()

                    y_flat, y_rsr, z, x_hat = self.forward(x_batch)
                    proj_loss = self.proj_error()

                    proj_loss.backward()
                    optimizer_proj.step()

                if self.all_alt:
                    optimizer_proj.zero_grad()

                    y_flat, y_rsr, z, x_hat = self.forward(x_batch)
                    pca_loss = self.pca_error(y_flat, y_rsr)

                    pca_loss.backward()
                    optimizer_proj.step()

            if (
                self.batch_show is not None
                and (epoch + 1) % self.batch_show == 0
            ):
                if x_val is not None:
                    with torch.no_grad():
                        x_val_t = torch.tensor(
                            x_val,
                            dtype=torch.float32,
                            device=device,
                        )

                        y_flat_val, y_rsr_val, z_val, x_hat_val = (
                            self.forward(x_val_t)
                        )

                        val_loss = self.reconstruction_error(
                            x_val_t,
                            x_hat_val,
                        )

                        print(
                            f"Epoch {epoch + 1}/{self.epoch_size} - "
                            f"Val Loss: {val_loss.item():.4f}"
                        )
                else:
                    print(
                        f"Epoch {epoch + 1}/{self.epoch_size} - "
                        f"Loss: {loss.item():.4f}"
                    )

    def get_latent(self, X, device="cuda"):
        self.to(device)

        with torch.no_grad():
            _, _, z, _ = self.forward(X)

        return z.cpu().numpy()

    def get_output(self, X, device="cuda"):
        self.to(device)

        with torch.no_grad():
            _, _, _, x_hat = self.forward(X)

        return x_hat.cpu().numpy()


class RSRAE(BaselineModel):

    def __init__(self, args):
        self.batch_size = args["batch_size"]
        args.pop("batch_size", None)

        self.model = CAE(**args)

    def train(self, X_train, device="cuda"):
        self.model.fit(
            X_train,
            self.batch_size,
            None,
            device,
        )

        return self.model

    def test(self, X_test, y_test, device="cuda"):
        with torch.no_grad():
            X_test = X_test.to(device)

            features = self.model.get_output(
                X_test,
                device=device,
            )

            flat_output = np.reshape(
                features,
                (features.shape[0], -1),
            )
            flat_input = np.reshape(
                X_test.cpu().numpy(),
                (X_test.shape[0], -1),
            )

            cosine_similarity = (
                np.sum(flat_output * flat_input, -1)
                / (np.linalg.norm(flat_output, axis=-1) + 1e-6)
                / (np.linalg.norm(flat_input, axis=-1) + 1e-6)
            )

            auc_rsrae, fpr95_rsrae, ap_rsrae = ev.evaluation(
                y_test,
                -cosine_similarity,
                verbose=False,
            )

            return auc_rsrae, fpr95_rsrae, ap_rsrae