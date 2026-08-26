import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

import Modelisation.Evaluation.evaluation as ev
from Modelisation.Baselines.baseline import BaselineModel


class SelfAttentionMultiContext(nn.Module):

    def __init__(self, hidden_size, attention_size, n_heads):
        super().__init__()

        self.W1 = nn.Linear(hidden_size, attention_size, bias=False)
        self.W2 = nn.Linear(attention_size, n_heads, bias=False)

    def forward(self, H, attention_mask):
        x = torch.tanh(self.W1(H))
        x = self.W2(x)

        mask = attention_mask.unsqueeze(-1)
        x = x.masked_fill(mask == 0, -1e9)

        A = F.softmax(x, dim=1)
        A = A.transpose(1, 2)

        M = torch.matmul(A, H)

        return M, A


class CVDDBERT(nn.Module):

    def __init__(
        self,
        bert_model,
        hidden_size=768,
        attention_size=100,
        n_heads=5,
        freeze_bert=True,
    ):
        super().__init__()

        self.bert = bert_model
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

        self.self_attention = SelfAttentionMultiContext(
            hidden_size=hidden_size,
            attention_size=attention_size,
            n_heads=n_heads,
        )

        self.C = nn.Parameter(torch.randn(n_heads, hidden_size))
        nn.init.xavier_uniform_(self.C)

        self.cosine_sim = nn.CosineSimilarity(dim=-1)

        # Temperature parameter, updated by the scheduler during training.
        self.alpha = 0.0

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )

        H = outputs.last_hidden_state

        M, A = self.self_attention(H, attention_mask)

        M = F.normalize(M, dim=-1)
        C = F.normalize(self.C, dim=-1)

        cosine_dists = 0.5 * (
            1 - self.cosine_sim(M, C.unsqueeze(0))
        )

        context_weights = F.softmax(
            -self.alpha * cosine_dists,
            dim=1,
        )

        return cosine_dists, context_weights, A


def cvdd_loss(cosine_dists, context_weights, C, lambda_p):
    loss_emp = torch.mean(
        torch.sum(context_weights * cosine_dists, dim=1)
    )

    r = C.size(0)
    I = torch.eye(r, device=C.device)
    CCT = torch.matmul(C, C.t())

    loss_reg = torch.mean((CCT - I) ** 2)

    return loss_emp + lambda_p * loss_reg


class AlphaScheduler:

    def __init__(self, milestones, values):
        self.milestones = milestones
        self.values = values
        self.idx = 0

    def step(self, epoch, model):
        if (
            self.idx < len(self.milestones)
            and epoch == self.milestones[self.idx]
        ):
            model.alpha = float(self.values[self.idx])
            self.idx += 1


def train_cvdd(
    model,
    dataloader,
    optimizer,
    alpha_scheduler,
    lambda_p,
    device,
    n_epochs,
):
    model.to(device)
    model.train()

    for epoch in range(n_epochs):
        alpha_scheduler.step(epoch, model)

        epoch_loss = 0.0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            optimizer.zero_grad()

            cosine_dists, context_weights, _ = model(
                input_ids,
                attention_mask,
            )

            loss = cvdd_loss(
                cosine_dists,
                context_weights,
                model.C,
                lambda_p,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % (n_epochs // 3) == 0:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.6f}")


def anomaly_score(model, dataloader, device, mode="best"):
    model.eval()
    scores = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            cosine_dists, _, _ = model(
                input_ids,
                attention_mask,
            )

            if mode == "mean":
                batch_scores = torch.mean(cosine_dists, dim=1)
            elif mode == "best":
                batch_scores, _ = torch.min(cosine_dists, dim=1)
            else:
                raise ValueError("mode must be 'mean' or 'best'")

            scores.append(batch_scores.cpu())

    return torch.cat(scores)


class CVDDDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": self.labels[idx],
        }


class CVDD(BaselineModel):

    def __init__(self, args):
        self.bert_name = args["bert_name"]

        self.tokenizer = AutoTokenizer.from_pretrained(self.bert_name)
        self.bert = AutoModel.from_pretrained(self.bert_name)

        self.hidden_size = args["hidden_size"]
        self.n_attention_heads = args["n_attention_heads"]
        self.attention_size = args["attention_size"]
        self.freeze_bert = args["freeze_bert"]

        self.model = CVDDBERT(
            bert_model=self.bert,
            hidden_size=self.hidden_size,
            n_heads=self.n_attention_heads,
            attention_size=self.attention_size,
            freeze_bert=self.freeze_bert,
        )

        self.alpha_scheduler = AlphaScheduler(
            milestones=[5, 40, 80],
            values=[0.0, 1.0, 2.0],
        )

        self.lr = args["lr"]
        self.weight_decay = args["weight_decay"]

        self.optimizer = Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        self.lambda_p = args["lambda_p"]
        self.n_epochs = args["n_epochs"]
        self.batch_size = args["batch_size"]
        self.device = args["device"]

    def train(self, data_train, column_="text"):
        train_dataset = CVDDDataset(
            data_train[column_],
            data_train["anomaly_class"],
            self.tokenizer,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        train_cvdd(
            model=self.model,
            dataloader=train_loader,
            optimizer=self.optimizer,
            alpha_scheduler=self.alpha_scheduler,
            lambda_p=self.lambda_p,
            device=self.device,
            n_epochs=self.n_epochs,
        )

    def test(self, data_test, column_="text"):
        test_dataset = CVDDDataset(
            data_test[column_],
            data_test["anomaly_class"],
            self.tokenizer,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        scores = anomaly_score(
            self.model,
            test_loader,
            self.device,
        ).numpy()

        y_test = np.array(
            [x["label"] for x in test_dataset]
        )

        auc, fpr95, ap = ev.evaluation(
            y_test,
            scores,
            verbose=True,
        )

        return auc, fpr95, ap