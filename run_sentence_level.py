"""
Textual Anomaly Detection - Sentence-level (SentenceBERT/mpnet/e5...) embeddings.

Modèles évalués : FLOCAT (flow matching), RSRAE, TCCM, FATE, LLM-based.
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
from datasets import concatenate_datasets

import data_loading as dl
import Data.embedding_encoder as embedding_encoder
from Modelisation.Baselines.RSRAE.model import RSRAE
from Modelisation.Baselines.TCCM.model import TCCM
from Modelisation.Flocat.flocat import flocat, flocatTrainer
from Modelisation.Baselines.FATE.model import FATE
from Modelisation.Baselines.LLM.model import LLMAnomalyDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Sentence-level AD experiments")

    parser.add_argument("--dataset_name", type=str, default="20newsgroups")
    parser.add_argument("--runall", action="store_true")

    args, _ = parser.parse_known_args()

    if not args.runall:
        parser.add_argument("--inlier_topic", type=str, default="computer")

    parser.add_argument("--type_tac", type=str, default="pantin", choices=["ruff", "pantin", "fate"])
    parser.add_argument("--nu", type=float, default=0.0)
    parser.add_argument("--nu_contamination", type=float, default=0.0)
    parser.add_argument("--nb_runs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--device", type=str, default=None)

    parser.add_argument("--model_name", type=str, default="all-mpnet-base-v2")
    parser.add_argument("--embedding_name", type=str, default="mpnet_embedding")
    parser.add_argument("--encoder_type", type=str, default="sentencebert")

    parser.add_argument("--llm_name", type=str, default="")

    parser.add_argument("--flocat", action="store_true")
    parser.add_argument("--rsrae", action="store_true")
    parser.add_argument("--tccm", action="store_true")
    parser.add_argument("--fate", action="store_true")
    parser.add_argument("--llm", action="store_true")

    return parser.parse_args()


def embed(dataset, encoder, text_column):
    return encoder.forward(dataset, text_column)


def main(args):
    device = dl.get_device(args.device)
    text_column = dl.get_text_column(args.dataset_name)
    inlier_topics = dl.get_inlier_topics(args.dataset_name, getattr(args, "inlier_topic", None), args.runall)

    print(f"\n<<<<<<<<<<<<<< {args.dataset_name} >>>>>>>>>>>>>>>>>>>\n")

    train_dataset, test_dataset = dl.load_dataset(args.dataset_name, args.batch_size)
    train_dataset, test_dataset = dl.preprocess_dataset(args.dataset_name, train_dataset, test_dataset)

    encoder = embedding_encoder.EmbeddingEncoder(
        args.model_name, args.embedding_name, args.encoder_type, device,
    )

    for i, inlier_topic in enumerate(inlier_topics):
        print(f"------------------------ {inlier_topic} ({i + 1}/{len(inlier_topics)}) -----------------------------")

        metrics = {
            name: {"auc": [], "fpr": [], "ap": [], "time": []}
            for name in ("flocat", "rsrae", "tccm", "fate", "llm")
            if getattr(args, name)
        }

        train_inlier, train_anomaly = dl.apply_tac(
            train_dataset, args.dataset_name, inlier_topic, args.type_tac, args.nu,
        )
        train_inlier = embed(train_inlier, encoder, text_column)
        train_anomaly = embed(train_anomaly, encoder, text_column)

        X_inlier = Tensor(train_inlier[args.embedding_name]).to(device)
        X_anom_for_train = Tensor(train_anomaly[args.embedding_name]).to(device) if args.nu > 0 else None
        print(X_inlier.shape)

        # FATE a besoin de son propre jeu d'anomalies "texte" (type_tac dédié).
        # NOTE: si tu veux garder le comportement d'origine (fichier séparé,
        # indépendant de --nu), ajuste ici le nu passé à apply_tac.
        train_anomaly_fate = None
        if args.fate:
            _, train_anomaly_fate = dl.apply_tac(
                train_dataset, args.dataset_name, inlier_topic, "fate", args.nu_contamination,
            )

        for n_run in range(1, args.nb_runs + 1):
            print(f"+++++++++++++++++++++ run : {n_run}/{args.nb_runs} +++++++++++++++++\n")

            test_inlier, test_anomaly = dl.apply_tac(
                test_dataset, args.dataset_name, inlier_topic, args.type_tac, args.nu,
            )
            test_inlier = embed(test_inlier, encoder, text_column)
            test_anomaly = embed(test_anomaly, encoder, text_column)

            _, y_test = dl.build_labeled_test_set(test_inlier, test_anomaly)
            X_test = torch.cat([
                Tensor(test_inlier[args.embedding_name]),
                Tensor(test_anomaly[args.embedding_name]),
            ]).to(device)
            print(X_test.shape)

            # ---------------- RSRAE ----------------
            if args.rsrae:
                rsrae_args = {
                    "input_dim": X_inlier.shape[1], "hidden_layer_sizes": (128, 64, 32), "intrinsic_size": 10,
                    "activation": nn.ReLU(), "norm_type": "l21", "loss_norm_type": "mse",
                    "if_rsr": True, "enforce_proj": True, "all_alt": True,
                    "learning_rate": 1e-3, "lambda1": 0.1, "lambda2": 0.1,
                    "epoch_size": 200, "batch_show": 50, "normalize": True,
                    "bn": False, "seed": 42, "batch_size": 128,
                }
                model = RSRAE(rsrae_args)
                t0 = time.time()
                src = torch.cat([X_inlier, X_anom_for_train]) if args.nu_contamination > 0 else X_inlier
                model.train(src, device)
                t1 = time.time()
                auc, fpr, ap = model.test(X_test, y_test)
                print(f"RSRAE --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["rsrae"][k].append(v)

            # ---------------- TCCM ----------------
            if args.tccm:
                tccm_args = {
                    "n_features": X_inlier.shape[1], "epochs": 30, "learning_rate": 1e-3,
                    "batch_size": 128, "device": device,
                }
                model = TCCM(tccm_args)
                t0 = time.time()
                src = torch.cat([X_inlier, X_anom_for_train]) if args.nu_contamination > 0 else X_inlier
                model.train(src)
                t1 = time.time()
                auc, fpr, ap = model.test(X_test, y_test)
                print(f"TCCM --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["tccm"][k].append(v)

            # ---------------- FATE ----------------
            if args.fate:
                fate_args = {
                    "model_name": args.model_name,
                    "device": device,
                    "batch_size": 512,
                    "n_epochs": 10,
                    "lr": 1e-3,
                    "include_regularization": True,
                    "top_k": 0.1,
                    "nb_shot": 5,
                    "train_inlier_text": train_inlier[text_column],
                    "train_anomaly_text": train_anomaly_fate[text_column],
                    "test_inlier_text": test_inlier[text_column],
                    "test_anomaly_text": test_anomaly[text_column],
                }
                model = FATE(fate_args)
                t0 = time.time()
                model.train()
                t1 = time.time()
                auc, fpr, ap = model.test()
                print(f"FATE --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["fate"][k].append(v)

            # ---------------- LLM ----------------
            if args.llm:
                detector = LLMAnomalyDetector(
                    normal_category=inlier_topic,
                    model_name=LLMAnomalyDetector.MODEL_OPTIONS[args.llm_name],
                    n_inlier_examples=3,
                    outlier_examples=None,
                    threshold=0.5,
                    verbose=True,
                )
                t0 = time.time()
                detector.train(train_inlier[text_column])
                test_texts = list(test_inlier[text_column]) + list(test_anomaly[text_column])
                auc, fpr, ap = detector.test(test_texts, y_test=y_test)
                t1 = time.time()
                print(f"LLM --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["llm"][k].append(v)

            # ---------------- FLOCAT (Flow Matching Transformers) ----------------
            if args.flocat:
                flocat_config = {
                    "latent_dim": X_inlier.shape[1], "hidden_dim": 128, "depth": 8, "n_heads": 8,
                    "freq_embed_size": 128, "lr": 1e-3, "weight_decay": 1e-5, "lambda_love": 0,
                    "epochs": 300, "lr_epochs": 120, "batch_size": args.batch_size,
                    "coef_var": 0.4, "target": "gaussian-neigh", "source": X_inlier,
                    "attentions_mask": None, "device": device,
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
                auc, fpr, ap = flocatformer.test(X_test, y_test, type="norm-centroid", n_steps=10)
                print(f"FLOCAT --> AUC: {auc:.4f} | FPR@95: {fpr:.4f} | AP: {ap:.4f}\n")
                for k, v in zip(("auc", "fpr", "ap", "time"), (auc, fpr, ap, t1 - t0)):
                    metrics["flocat"][k].append(v)

if __name__ == "__main__":
    main(parse_args())

# python3 run_sentence_level.py \
#     --dataset_name "reuters" \
#     --inlier_topic "acq" \
#     --type_tac "ruff" \
#     --nu 0.1 \
#     --model_name "all-mpnet-base-v2" \
#     --embedding_name "mpnet_embedding" \
#     --nb_runs 5 \
#     --flocat --rsrae --tccm --fate --llm --llm_name "..."


# python3 run_sentence_level.py \
#     --dataset_name "reuters" \
#     --inlier_topic "acq" \
#     --type_tac "ruff" \
#     --nu 0.1 \
#     --model_name "all-mpnet-base-v2" \
#     --embedding_name "mpnet_embedding" \
#     --nb_runs 2 \
#     --tccm