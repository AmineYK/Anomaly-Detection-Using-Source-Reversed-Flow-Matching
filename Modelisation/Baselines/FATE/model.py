import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader, Sampler, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoTokenizer

import Modelisation.Evaluation.evaluation as ev
from Modelisation.Baselines.baseline import BaselineModel


class BalancedBatchSampler(Sampler):

    def __init__(self, dataset, batch_size):
        super().__init__(dataset)

        self.dataset = dataset
        self.batch_size = batch_size
        self.anomalies = True
        self.steps_per_epoch = len(dataset) // batch_size

        self.normal_generator = self.random_generator(
            self.dataset.normal_idx
        )
        self.outlier_generator = self.random_generator(
            self.dataset.outlier_idx
        )

        if self.anomalies:
            self.n_normal = batch_size // 2
            self.n_outlier = batch_size - self.n_normal
        else:
            self.n_normal = batch_size
            self.n_outlier = 0

    @staticmethod
    def random_generator(idx_list):
        while True:
            random_list = np.random.permutation(idx_list)

            for idx in random_list:
                yield idx

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        for _ in range(self.steps_per_epoch):
            batch = []

            for _ in range(self.n_normal):
                batch.append(next(self.normal_generator))

            for _ in range(self.n_outlier):
                batch.append(next(self.outlier_generator))

            yield batch


def worker_init_fn_seed(worker_id):
    seed = 10 + worker_id
    np.random.seed(seed)


class CustomDataset(TensorDataset):

    def __init__(
        self,
        embeddings,
        labels,
        normal_idx=None,
        outlier_idx=None,
    ):
        super().__init__(embeddings, labels)

        self.normal_idx = normal_idx
        self.outlier_idx = outlier_idx


class DeviationLoss(nn.Module):

    def __init__(self, confidence_margin=5.0, n_ref=5000):
        super().__init__()

        self.confidence_margin = confidence_margin

        ref = torch.normal(
            mean=0.0,
            std=torch.ones(n_ref),
        )

        self.mu_ref = ref.mean().item()
        self.sigma_ref = ref.std().item()

    def forward(self, y_pred, y_true):
        dev = (
            y_pred - self.mu_ref
        ) / self.sigma_ref

        inlier_loss = torch.abs(dev)
        outlier_loss = torch.clamp(
            self.confidence_margin - dev,
            min=0.0,
        )

        return torch.mean(
            (1 - y_true) * inlier_loss
            + y_true * outlier_loss
        )


class SBERTWithAttention(nn.Module):

    def __init__(
        self,
        model_name="all-distilroberta-v1",
        attention_size=150,
        num_heads=5,
        top_k=0.1,
    ):
        super().__init__()

        self.sbert = SentenceTransformer(model_name)
        self.hidden_size = (
            self.sbert.get_sentence_embedding_dimension()
        )
        self.attention_size = attention_size
        self.num_heads = num_heads
        self.top_k = top_k

        self.W1 = nn.Linear(
            self.hidden_size,
            self.attention_size,
            bias=False,
        )
        self.W2 = nn.Linear(
            self.attention_size,
            self.num_heads,
            bias=False,
        )

    def forward(self, features):
        sbert_output = self.sbert(features)
        hidden_state = sbert_output["token_embeddings"]

        t = torch.tanh(
            self.W1(hidden_state)
        )
        t = F.softmax(
            self.W2(t),
            dim=1,
        )

        attention_matrix = t.transpose(1, 2)

        outputs = attention_matrix @ hidden_state
        outputs = torch.flatten(
            outputs,
            start_dim=1,
        )

        topk = max(
            int(outputs.size(1) * self.top_k),
            1,
        )

        outputs = torch.topk(
            torch.abs(outputs),
            topk,
            dim=1,
        )[0]

        outputs = outputs.mean(dim=1)
        outputs = outputs.float()

        return outputs, attention_matrix


class FATETrainer:

    def __init__(
        self,
        model_name,
        inlier_sentences,
        anom_sentences,
        test_inlier_sentences,
        test_anom_sentences,
        device=None,
        batch_size=16,
        num_epochs=10,
        include_regularization=True,
        top_k=0.1,
    ):
        self.model_name = model_name
        self.device = (
            device
            or torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.include_regularization = include_regularization

        model_path = f"sentence-transformers/{model_name}"
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )

        self.model = SBERTWithAttention(
            model_name=self.model_name,
            top_k=top_k,
        ).to(self.device)

        self.criterion = DeviationLoss()

        (
            self.train_dataset,
            self.test_inlier_dataset,
            self.test_anom_dataset,
        ) = self.prepare_datasets(
            inlier_sentences,
            anom_sentences,
            test_inlier_sentences,
            test_anom_sentences,
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_sampler=BalancedBatchSampler(
                self.train_dataset,
                batch_size,
            ),
            worker_init_fn=worker_init_fn_seed,
        )

        self.test_inlier_loader = DataLoader(
            self.test_inlier_dataset,
            sampler=SequentialSampler(
                self.test_inlier_dataset
            ),
            batch_size=batch_size,
        )

        self.test_anom_loader = DataLoader(
            self.test_anom_dataset,
            sampler=SequentialSampler(
                self.test_anom_dataset
            ),
            batch_size=batch_size,
        )

    def prepare_datasets(
        self,
        inliers,
        anomalies,
        test_inliers,
        test_anom,
    ):
        def encode(sentences):
            tokenized = self.tokenizer(
                sentences,
                padding="max_length",
                truncation=True,
                max_length=128,
                return_tensors="pt",
            )

            return (
                tokenized["input_ids"],
                tokenized["attention_mask"],
            )

        input_ids_inlier, att_inlier = encode(inliers)
        input_ids_anom, att_anom = encode(anomalies)

        labels = torch.cat(
            [
                torch.zeros(len(inliers)),
                torch.ones(len(anomalies)),
            ]
        )

        normal_idx = torch.arange(len(inliers))
        outlier_idx = torch.arange(
            len(inliers),
            len(inliers) + len(anomalies),
        )

        train_dataset = CustomDataset(
            torch.cat(
                [input_ids_inlier, input_ids_anom]
            ),
            torch.cat(
                [att_inlier, att_anom]
            ),
            labels,
            normal_idx,
            outlier_idx,
        )

        def make_test_dataset(sentences, label_val):
            input_ids, att_mask = encode(sentences)

            labels = torch.full(
                (len(sentences),),
                label_val,
            )

            return CustomDataset(
                input_ids,
                att_mask,
                labels,
            )

        test_inlier_dataset = make_test_dataset(
            test_inliers,
            0,
        )
        test_anom_dataset = make_test_dataset(
            test_anom,
            1,
        )

        return (
            train_dataset,
            test_inlier_dataset,
            test_anom_dataset,
        )

    def train(self, lr=1e-6):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        for epoch in range(self.num_epochs):
            self.model.train()

            for input_ids, attention_mask, labels in tqdm(
                self.train_loader
            ):
                optimizer.zero_grad()

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(
                    self.device
                )
                labels = labels.to(self.device).float()

                features = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                outputs, attention = self.model(
                    features
                )

                identity_mat = torch.eye(
                    attention.size(1),
                    device=self.device,
                )

                loss_reg = torch.mean(
                    (
                        attention
                        @ attention.transpose(1, 2)
                        - identity_mat
                    ) ** 2
                )

                loss = self.criterion(
                    outputs,
                    labels,
                )

                if self.include_regularization:
                    loss += loss_reg

                loss.backward()
                optimizer.step()

            print(
                f"Epoch {epoch + 1}/{self.num_epochs}, "
                f"Loss: {loss.item():.4f}"
            )

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()

        def compute_scores(loader):
            scores = []

            for input_ids, attention_mask, labels in loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(
                    self.device
                )

                features = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                }

                outputs, _ = self.model(features)
                scores.append(
                    outputs.cpu().numpy()
                )

            return np.concatenate(scores)

        inlier_scores = compute_scores(
            self.test_inlier_loader
        )
        anom_scores = compute_scores(
            self.test_anom_loader
        )

        gt = np.concatenate(
            [
                np.zeros_like(inlier_scores),
                np.ones_like(anom_scores),
            ]
        )
        preds = np.concatenate(
            [inlier_scores, anom_scores]
        )

        auc, fpr95, ap = ev.evaluation(
            gt,
            preds,
            verbose=False,
        )

        print(
            f"AUC: {auc:.4f}, "
            f"AP: {ap:.4f}, "
            f"FPR95: {fpr95:.4f}"
        )

        return auc, fpr95, ap


class FATEModel(BaselineModel):

    def __init__(self, args):
        self.model_name = args["model_name"]
        self.device = args["device"]
        self.batch_size = args["batch_size"]
        self.n_epochs = args["n_epochs"]
        self.include_regularization = args[
            "include_regularization"
        ]
        self.top_k = args["top_k"]
        self.lr = args["lr"]
        self.nb_shot = args["nb_shot"]

        self.train_inlier_text = args[
            "train_inlier_text"
        ]
        self.train_anomaly_text = args[
            "train_anomaly_text"
        ]
        self.test_inlier_text = args[
            "test_inlier_text"
        ]
        self.test_anomaly_text = args[
            "test_anomaly_text"
        ]

        indices = np.random.randint(
            0,
            len(self.train_anomaly_text),
            self.nb_shot,
        )

        self.trainer = FATETrainer(
            model_name=self.model_name,
            inlier_sentences=self.train_inlier_text,
            anom_sentences=np.array(
                self.train_anomaly_text
            )[indices].tolist(),
            test_inlier_sentences=self.test_inlier_text,
            test_anom_sentences=self.test_anomaly_text,
            batch_size=self.batch_size,
            num_epochs=self.n_epochs,
            include_regularization=self.include_regularization,
            top_k=self.top_k,
        )

    def train(self):
        self.trainer.train(self.lr)

    def test(self):
        auc, fpr95, ap = self.trainer.evaluate()

        return auc, fpr95, ap