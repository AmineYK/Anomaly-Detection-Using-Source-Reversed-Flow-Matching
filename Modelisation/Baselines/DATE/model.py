import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    BertConfig,
    BertForMaskedLM,
    BertModel,
    BertTokenizerFast,
    RobertaConfig,
    RobertaForMaskedLM,
    RobertaModel,
    RobertaTokenizerFast,
)

import Modelisation.Evaluation.evaluation as ev
from Modelisation.Baselines.DATE.utils import (
    apply_mask_safe,
    corrupt_input_random,
    date_anomaly_score,
    generate_mask_patterns
)
from Modelisation.Baselines.baseline import BaselineModel


class DateGenerator(nn.Module):

    def __init__(self, which_config, vocab_size):
        super().__init__()

        if which_config == "bert":
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_act="gelu",
                max_position_embeddings=512,
            )
            self.model = BertForMaskedLM(config)

        elif which_config == "roberta":
            config = RobertaConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                max_position_embeddings=512,
            )
            self.model = RobertaForMaskedLM(config)

        else:
            raise ValueError(
                f"Unknown config: {which_config}. "
                "Expected 'bert' or 'roberta'."
            )

    def forward(self, input_ids, attention_mask):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        return outputs.logits


class DateDiscriminator(nn.Module):

    def __init__(self, which_config, vocab_size, K):
        super().__init__()

        if which_config == "bert":
            config = BertConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                hidden_act="gelu",
                max_position_embeddings=512,
            )
            self.encoder = BertModel(config)

        elif which_config == "roberta":
            config = RobertaConfig(
                vocab_size=vocab_size,
                hidden_size=256,
                num_hidden_layers=4,
                num_attention_heads=4,
                intermediate_size=1024,
                max_position_embeddings=512,
            )
            self.encoder = RobertaModel(config)

        else:
            raise ValueError(
                f"Unknown config: {which_config}. "
                "Expected 'bert' or 'roberta'."
            )

        self.rtd_head = nn.Linear(config.hidden_size, 1)
        self.rmd_head = nn.Linear(config.hidden_size, K)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        hidden = outputs.last_hidden_state
        cls = hidden[:, 0]

        rtd_logits = self.rtd_head(hidden).squeeze(-1)
        rmd_logits = self.rmd_head(cls)

        return rtd_logits, rmd_logits


class DATEDataset(Dataset):

    def __init__(
        self,
        texts,
        labels=None,
        tokenizer=None,
        max_len=498,
    ):
        self.texts = texts
        self.labels = (
            labels if labels is not None else [0] * len(texts)
        )
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="pt",
        )

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return (
            self.encodings["input_ids"][idx],
            self.encodings["attention_mask"][idx],
            torch.tensor(
                self.labels[idx],
                dtype=torch.long,
            ),
        )


class DATE(BaselineModel):

    def __init__(self, args):
        self.which_config = args["which_config"]
        self.encoder_name = args["encoder_name"]

        if self.which_config == "bert":
            tokenizer_class = BertTokenizerFast
        elif self.which_config == "roberta":
            tokenizer_class = RobertaTokenizerFast
        else:
            raise ValueError(
                f"Unknown config: {self.which_config}. "
                "Expected 'bert' or 'roberta'."
            )

        self.tokenizer = tokenizer_class.from_pretrained(
            self.encoder_name
        )
        self.mask_token_id = self.tokenizer.mask_token_id

        self.device = args["device"]
        self.K = args["K"]
        self.vocab_size = len(self.tokenizer)

        self.generator = DateGenerator(
            self.which_config,
            self.vocab_size,
        ).to(self.device)

        self.discriminator = DateDiscriminator(
            self.which_config,
            self.vocab_size,
            self.K,
        ).to(self.device)

        self.lr = args["lr"]
        self.weight_decay = args["weight_decay"]

        self.optimizer = torch.optim.AdamW(
            list(self.generator.parameters())
            + list(self.discriminator.parameters()),
            lr=self.lr,
            weight_decay=self.weight_decay,
            amsgrad=True,
        )

        self.seq_len = args["seq_len"]
        self.ratio = args["ratio"]

        self.mask_patterns = generate_mask_patterns(
            K=self.K,
            seq_len=self.seq_len,
            ratio=self.ratio,
        ).to(self.device)

        self.n_epochs = args["n_epochs"]
        self.batch_size = args["batch_size"]

    def train(self, data_train, column="text"):
        train_texts = data_train[column]

        train_ds = DATEDataset(
            train_texts,
            None,
            self.tokenizer,
            max_len=self.seq_len,
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=self.batch_size,
            shuffle=True,
        )

        self.discriminator.train()

        print(
            f"Training sur {len(train_texts)} samples, "
            f"{len(train_loader)} batches"
        )
        print(
            f"K={self.K} masks, "
            f"ratio={self.ratio}, "
            f"seq_len={self.seq_len}"
        )

        for epoch in range(self.n_epochs):
            total_loss = 0.0
            total_rmd = 0.0
            total_rtd = 0.0
            n_masked_tokens = 0

            for input_ids, attention_mask, _ in train_loader:
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)

                k = torch.randint(
                    0,
                    self.K,
                    (1,),
                ).item()

                mask = self.mask_patterns[k]

                masked_ids, actual_mask = apply_mask_safe(
                    input_ids,
                    mask,
                    self.tokenizer,
                )

                n_masked_tokens += actual_mask.sum().item()

                corrupted_ids = corrupt_input_random(
                    masked_ids,
                    self.tokenizer.mask_token_id,
                    self.vocab_size,
                )

                rtd_labels = (corrupted_ids != input_ids).long()

                rtd_labels[
                    input_ids == self.tokenizer.cls_token_id
                ] = -100
                rtd_labels[
                    input_ids == self.tokenizer.pad_token_id
                ] = -100
                rtd_labels[attention_mask == 0] = -100

                rmd_labels = torch.full(
                    (input_ids.size(0),),
                    k,
                    device=self.device,
                    dtype=torch.long,
                )

                rtd_logits, rmd_logits = self.discriminator(
                    corrupted_ids,
                    attention_mask,
                )

                ce = nn.CrossEntropyLoss(
                    ignore_index=-100
                )
                bce = nn.BCEWithLogitsLoss(
                    reduction="none"
                )

                valid = rtd_labels != -100

                if valid.sum() > 0:
                    loss_rtd = bce(
                        rtd_logits[valid].float(),
                        rtd_labels[valid].float(),
                    ).mean()
                else:
                    loss_rtd = torch.tensor(
                        0.0,
                        device=self.device,
                    )

                loss_rmd = ce(
                    rmd_logits,
                    rmd_labels,
                )

                mu = 100.0
                lambda_rtd = 50.0

                loss = (
                    mu * loss_rmd
                    + lambda_rtd * loss_rtd
                )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.discriminator.parameters(),
                    max_norm=1.0,
                )

                self.optimizer.step()

                total_loss += loss.item()
                total_rmd += loss_rmd.item()
                total_rtd += loss_rtd.item()

            avg_loss = total_loss / len(train_loader)
            avg_rmd = total_rmd / len(train_loader)
            avg_rtd = total_rtd / len(train_loader)

            avg_masked = (
                n_masked_tokens
                / (
                    len(train_loader)
                    * self.batch_size
                    * self.seq_len
                )
            )

            if epoch % max(1, self.n_epochs // 5) == 0:
                print(
                    f"Epoch {epoch + 1}/{self.n_epochs} - "
                    f"Loss: {avg_loss:.2f} "
                    f"(RMD: {avg_rmd:.4f}, RTD: {avg_rtd:.4f}) "
                    f"| Masked: {avg_masked:.2%}"
                )

    def test(self, data_test, column="text"):
        test_texts = data_test[column]
        test_labels = data_test["anomaly_class"]

        test_ds = DATEDataset(
            test_texts,
            test_labels,
            self.tokenizer,
        )

        test_loader = DataLoader(
            test_ds,
            batch_size=64,
        )

        scores, _ = date_anomaly_score(
            self.discriminator,
            test_loader,
            self.tokenizer,
            self.device,
        )

        test_scores = -scores

        auc, fpr95, ap = ev.evaluation(
            test_labels,
            test_scores,
            verbose=False,
        )

        return auc, fpr95, ap