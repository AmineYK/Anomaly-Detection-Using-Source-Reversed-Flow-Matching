"""
Textual Anomaly Detection - Token-level (RoBERTa/ModernBERT/Qwen/Mistral) embeddings.

Modèles évalués : FLOCAT (flow matching), TCCM, RSRAE, DATE, CVDD.
Le chargement / preprocessing / contamination (TAC) passe désormais par
data_loading.py au lieu de fichiers locaux pré-calculés.
"""

import argparse
import logging
import time

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

import data_loading as dl
from Data.utils import encode_tokens
from Modelisation.Baselines.RSRAE.model import RSRAE
from Modelisation.Baselines.TCCM.model import TCCM
from Modelisation.Baselines.CVDD.model import CVDD
from Modelisation.Baselines.DATE.model import DATE
from Modelisation.Flocat.flocat import flocat, flocatTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN_MODEL_CONFIGS = {
    "roberta": {"model_name": "roberta-base", "model_type": "encoder"},
    "qwen": {"model_name": "Qwen/Qwen2.5-0.5B", "model_type": "decoder"}
}


def parse_args():
    parser = argparse.ArgumentParser(description="Token-level AD experiments")

    parser.add_argument("--dataset_name", type=str, default="20newsgroups")
    parser.add_argument("--runall", action="store_true")

    args, _ = parser.parse_known_args()

    if not args.runall:
        parser.add_argument("--inlier_topic", type=str, default="computer")

    parser.add_argument("--type_tac", type=str, default="pantin", choices=["ruff", "pantin", "fate"])
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--nu_contamination", type=float, default=0.0)
    parser.add_argument("--nb_runs", type=int, default=5)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument(
        "--type_emb", type=str, default="roberta", choices=list(TOKEN_MODEL_CONFIGS.keys()),
    )

    parser.add_argument("--flocat", action="store_true")
    parser.add_argument("--tccm", action="store_true")
    parser.add_argument("--rsrae", action="store_true")
    parser.add_argument("--date", action="store_true")
    parser.add_argument("--cvdd", action="store_true")

    return parser.parse_args()


def main(args):
    device = dl.get_device(args.device)
    text_column = dl.get_text_column(args.dataset_name)
    inlier_topics = dl.get_inlier_topics(args.dataset_name, getattr(args, "inlier_topic", None), args.runall)

    print(f"\n<<<<<<<<<<<<<< {args.dataset_name} >>>>>>>>>>>>>>>>>>>\n")
    print(f"Text column : '{text_column}'")

    cfg = TOKEN_MODEL_CONFIGS[args.type_emb]
    print(f"Loading token model : {cfg['model_name']} (type={cfg['model_type']})")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    token_model = AutoModel.from_pretrained(cfg["model_name"], torch_dtype=torch.float32).to(device)
    token_model.eval()

    train_dataset, test_dataset = dl.load_dataset(args.dataset_name, args.batch_size)
    train_dataset, test_dataset = dl.preprocess_dataset(args.dataset_name, train_dataset, test_dataset)

    for i, inlier_topic in enumerate(inlier_topics):
        print(f"------------------------ {inlier_topic} ({i + 1}/{len(inlier_topics)}) -----------------------------")

        metrics = {
            name: {"auc": [], "fpr": [], "ap": [], "time": []}
            for name in ("flocat", "tccm", "rsrae", "date", "cvdd")
            if getattr(args, name)
        }

        train_inlier, train_anomaly = dl.apply_tac(
            train_dataset, args.dataset_name, inlier_topic, args.type_tac, args.nu,
        )

        X_inlier, _, attentions_train_mask = encode_tokens(
            token_model, tokenizer, train_inlier[text_column], device,
            batch_size=64, max_length=args.seq_len, model_type=cfg["model_type"],
        )
        print(X_inlier.shape)

        X_anom_for_train = None
        if args.nu_contamination > 0:
            X_anom_for_train, _, _ = encode_tokens(
                token_model, tokenizer, train_anomaly[text_column], device,
                batch_size=64, max_length=args.seq_len, model_type=cfg["model_type"],
            )

        for n_run in range(1, args.nb_runs + 1):
            print(f"+++++++++++++++++++++ run : {n_run}/{args.nb_runs} +++++++++++++++++\n")

            test_inlier, test_anomaly = dl.apply_tac(
                test_dataset, args.dataset_name, inlier_topic, args.type_tac, args.nu,
            )

            _, y_test = dl.build_labeled_test_set(test_inlier, test_anomaly)

            X_test_inlier, _, mask_test_inlier = encode_tokens(
                token_model, tokenizer, test_inlier[text_column], device,
                batch_size=64, max_length=args.seq_len, model_type=cfg["model_type"],
            )
            X_test_anomaly, _, mask_test_anomaly = encode_tokens(
                token_model, tokenizer, test_anomaly[text_column], device,
                batch_size=64, max_length=args.seq_len, model_type=cfg["model_type"],
            )
            X_test = torch.cat([X_test_inlier, X_test_anomaly])
            attentions_test_mask = torch.cat([mask_test_inlier, mask_test_anomaly])
            print(X_test.shape)

            # ---------------- RSRAE ----------------
            if args.rsrae:
                rsrae_args = {
                    "input_dim": X_inlier.shape[2], "hidden_layer_sizes": (128, 64, 32), "intrinsic_size": 10,
                    "activation": nn.ReLU(), "norm_type": "l21", "loss_norm_type": "mse",
                    "if_rsr": True, "enforce_proj": True, "all_alt": True,
                    "learning_rate": 1e-4, "lambda1": 0.1, "lambda2": 0.1,
                    "epoch_size": 50, "batch_show": 50, "normalize": True,
                    "bn": False, "seed": 42, "batch_size": X_inlier.shape[0] // 10,
                }
                model = RSRAE(rsrae_args)
                t0 = time.time()
                if args.nu_contamination > 0:
                    src = torch.cat([X_inlier.mean(dim=1), X_anom_for_train.mean(dim=1)])
                else:
                    src = X_inlier.mean(dim=1)
                model.train(src, device)
                t1 = time.time()
                auc, fpr, ap = model.test(X_test.mean(dim=1), y_test, device)
                print(f"RSRAE --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["rsrae"][k].append(v)

            # ---------------- TCCM ----------------
            if args.tccm:
                tccm_args = {
                    "n_features": X_inlier.shape[2], "epochs": 50, "learning_rate": 1e-3,
                    "batch_size": 32, "device": device,
                }
                model = TCCM(tccm_args)
                t0 = time.time()
                if args.nu_contamination > 0:
                    src = torch.cat([X_inlier, X_anom_for_train])
                else:
                    src = X_inlier.mean(dim=1)
                model.train(src)
                t1 = time.time()
                auc, fpr, ap = model.test(X_test.mean(dim=1), y_test)
                print(f"TCCM --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["tccm"][k].append(v)

            # ---------------- CVDD ----------------
            if args.cvdd:
                cvdd_args = {
                    "bert_name": cfg["model_name"], "hidden_size": X_inlier.shape[-1],
                    "n_attention_heads": 10, "attention_size": 64, "freeze_bert": True,
                    "lr": 1e-3, "weight_decay": 0, "lambda_p": 0.1,
                    "n_epochs": 20, "batch_size": 32, "device": device,
                }
                model = CVDD(cvdd_args)
                t0 = time.time()
                if args.nu_contamination > 0:
                    from datasets import concatenate_datasets
                    model.train(concatenate_datasets([train_inlier, train_anomaly]))
                else:
                    model.train(train_inlier, text_column)
                t1 = time.time()
                test_combined, _ = dl.build_labeled_test_set(test_inlier, test_anomaly)
                auc, fpr, ap = model.test(test_combined, text_column)
                print(f"CVDD --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["cvdd"][k].append(v)

            # ---------------- DATE ----------------
            if args.date:
                date_args = {
                    "which_config": args.type_emb, "encoder_name": cfg["model_name"],
                    "K": 20, "lr": 1e-3, "weight_decay": 1e-4, "seq_len": 64,
                    "ratio": 0.50, "n_epochs": 5, "batch_size": 32, "device": device,
                }
                model = DATE(date_args)
                t0 = time.time()
                if args.nu_contamination > 0:
                    from datasets import concatenate_datasets
                    model.train(concatenate_datasets([train_inlier, train_anomaly]))
                else:
                    model.train(train_inlier, text_column)
                t1 = time.time()
                test_combined, _ = dl.build_labeled_test_set(test_inlier, test_anomaly)
                auc, fpr, ap = model.test(test_combined, text_column)
                print(f"DATE --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["date"][k].append(v)

            # ---------------- FLOCAT ----------------
            if args.flocat:
                flocat_config = {
                    "latent_dim": X_inlier.shape[-1], "hidden_dim": 64, "depth": 4, "n_heads": 4,
                    "freq_embed_size": 128, "lr": 1e-3, "weight_decay": 1e-2, "lambda_love": 1e-4,
                    "epochs": 10, "lr_epochs": 3, "batch_size": args.batch_size,
                    "coef_var": 1.0, "target": "gaussian-neigh", "source": X_inlier,
                    "attentions_mask": attentions_train_mask, "device": device,
                }
                flow_model = flocat(
                    latent_dim=flocat_config["latent_dim"],
                    hidden_dim=flocat_config["hidden_dim"],
                    depth=flocat_config["depth"],
                    n_heads=flocat_config["n_heads"],
                ).to(device)
                flocatformer = flocatTrainer(flow_model, flocat_config)

                t0 = time.time()
                flocatformer.train(True)
                t1 = time.time()
                auc, fpr, ap = flocatformer.test(
                    X_test, y_test, attentions_test_mask, type="norm-centroid", n_steps=10,
                )
                print(f"FLOCAT --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["flocat"][k].append(v)


if __name__ == "__main__":
    main(parse_args())

# python3 run_token_level.py \
#     --dataset_name "reuters" \
#     --inlier_topic "acq" \
#     --type_tac "ruff" \
#     --nu 0.1 \
#     --type_emb "roberta" \
#     --seq_len 128 \
#     --nb_runs 5 \
#     --flocat --tccm --rsrae --date --cvdd


# python3 run_token_level.py \
#     --dataset_name "reuters" \
#     --inlier_topic "acq" \
#     --type_tac "ruff" \
#     --nu 0.1 \
#     --type_emb "roberta" \
#     --seq_len 64 \
#     --nb_runs 2 \
#     --tccm