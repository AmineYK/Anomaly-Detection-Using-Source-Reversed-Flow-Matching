import argparse
import logging
import time

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from datasets import concatenate_datasets

# ============================================================
# Dataset preparation
# ============================================================

from Data.run_dataset_create import get_datasets


# ============================================================
# Models
# ============================================================

from Modelisation.Baselines.RSRAE.model import RSRAE
from Modelisation.Baselines.TCCM.model import TCCM
from Modelisation.Baselines.CVDD.model import CVDD
from Modelisation.Baselines.FATE.model import FATE
from Modelisation.Baselines.DATE.model import DATE

from Modelisation.flocat.flocat import (
    flocat,
    flocatTrainer,
)

import Modelisation.Evaluation.evaluation as ev

# TODO
# from utils import save_results


# ============================================================
# Logger
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ============================================================
# Arguments
# ============================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Textual Anomaly Detection - Model Experiments"
    )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset to process.",
    )

    parser.add_argument(
        "--inlier_topic",
        type=str,
        default=None,
        help="Inlier topic.",
    )

    # --------------------------------------------------------
    # TAC
    # --------------------------------------------------------

    parser.add_argument(
        "--type_tac",
        type=str,
        default="pantin",
        choices=[
            "ruff",
            "pantin",
            "fate",
        ],
        help="Textual Anomaly Contamination type.",
    )

    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Anomaly contamination rate.",
    )

    # --------------------------------------------------------
    # Embedding
    # --------------------------------------------------------

    parser.add_argument(
        "--model_name",
        type=str,
        default="all-mpnet-base-v2",
        help="Sentence embedding model.",
    )

    parser.add_argument(
        "--embedding_name",
        type=str,
        default="mpnet_embedding",
        help="Embedding column.",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="sentencebert",
        help="Embedding encoder type.",
    )

    # --------------------------------------------------------
    # Runs
    # --------------------------------------------------------

    parser.add_argument(
        "--nb_runs",
        type=int,
        default=5,
        help="Number of test runs.",
    )

    # --------------------------------------------------------
    # General
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size.",
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device.",
    )

    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    parser.add_argument(
        "--ocsvm",
        action="store_true",
        help="Run OCSVM.",
    )

    parser.add_argument(
        "--rsrae",
        action="store_true",
        help="Run RSRAE.",
    )

    parser.add_argument(
        "--ae",
        action="store_true",
        help="Run AutoEncoder.",
    )

    parser.add_argument(
        "--tccm",
        action="store_true",
        help="Run TCCM.",
    )

    parser.add_argument(
        "--cvdd",
        action="store_true",
        help="Run CVDD.",
    )

    parser.add_argument(
        "--date",
        action="store_true",
        help="Run DATE.",
    )

    parser.add_argument(
        "--fate",
        action="store_true",
        help="Run FATE.",
    )

    parser.add_argument(
        "--fm_trans",
        action="store_true",
        help="Run Flow Matching Transformer.",
    )

    return parser.parse_args()


# ============================================================
# Device
# ============================================================

def get_device(device_arg):

    if device_arg is not None:
        return torch.device(device_arg)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


# ============================================================
# Evaluation storage
# ============================================================

def initialize_results(args):

    results = {}

    if args.ocsvm:
        results["ocsvm"] = {
            "auc": [],
            "fpr": [],
            "ap": [],
            "time": [],
        }

    if args.rsrae:
        results["rsrae"] = {
            "auc": [],
            "fpr": [],
            "ap": [],
            "time": [],
        }

    if args.ae:
        results["ae"] = {
            "auc": [],
            "fpr": [],
            "ap": [],
            "time": [],
        }

    if args.tccm:
        results["tccm"] = {
            "auc": [],
            "fpr": [],
            "ap": [],
            "time": [],
        }

    if args.cvdd:
        results["cvdd"] = {
            "auc": [],
            "fpr": [],
            "ap": [],
            "time": [],
        }

    if args.date:
        results["date"] = {
            "auc": [],
            "fpr": [],
            "ap": [],
            "time": [],
        }

    if args.fate:
        results["fate"] = {
            "auc": [],
            "fpr": [],
            "ap": [],
            "time": [],
        }

    if args.fm_trans:
        results["fm_trans"] = {
            "auc": [],
            "fpr": [],
            "ap": [],
            "time": [],
        }

    return results


# ============================================================
# Prepare test data
# ============================================================

def prepare_test_data(
    test_inlier,
    test_anomaly,
):

    data_test = concatenate_datasets(
        [
            test_inlier,
            test_anomaly,
        ]
    )

    y_test = np.concatenate(
        [
            np.zeros(len(test_inlier)),
            np.ones(len(test_anomaly)),
        ]
    )

    return data_test, y_test


# ============================================================
# OCSVM
# ============================================================

def run_ocsvm(
    X_inlier,
    X_anomaly,
    X_test,
    y_test,
    device,
):

    logger.info("Running OCSVM...")

    ocsvm_args = {
        "nu": 0.1,
        "kernel": "rbf",
        "gamma": "scale",
    }

    model = OCSVM(ocsvm_args)

    start = time.time()

    X_train = torch.concatenate(
        [
            X_inlier.mean(dim=1),
            X_anomaly.mean(dim=1),
        ]
    ).cpu()

    model.train(X_train)

    train_time = time.time() - start

    auc, fpr95, ap = model.test(
        X_test.mean(dim=1).cpu(),
        y_test,
    )

    logger.info(
        f"OCSVM --> "
        f"AUC: {auc:.4f} | "
        f"FPR@95: {fpr95:.4f} | "
        f"AP: {ap:.4f}"
    )

    return auc, fpr95, ap, train_time


# ============================================================
# RSRAE
# ============================================================

def run_rsrae(
    X_inlier,
    X_anomaly,
    X_test,
    y_test,
    device,
):

    logger.info("Running RSRAE...")

    rsrae_args = {
        "input_dim": X_inlier.shape[1],
        "hidden_layer_sizes": (128, 64, 32),
        "intrinsic_size": 10,
        "activation": nn.ReLU(),
        "norm_type": "l21",
        "loss_norm_type": "mse",
        "if_rsr": True,
        "enforce_proj": True,
        "all_alt": True,
        "learning_rate": 1e-4,
        "lambda1": 0.1,
        "lambda2": 0.1,
        "epoch_size": 50,
        "batch_show": 50,
        "normalize": True,
        "bn": False,
        "seed": 42,
        "batch_size": X_inlier.shape[0] // 10,
    }

    model = RSRAE(rsrae_args)

    start = time.time()

    X_train = torch.concatenate(
        [
            X_inlier,
            X_anomaly,
        ]
    )

    model.train(
        X_train,
        device,
    )

    train_time = time.time() - start

    auc, fpr95, ap = model.test(
        X_test,
        y_test,
        device,
    )

    logger.info(
        f"RSRAE --> "
        f"AUC: {auc:.4f} | "
        f"FPR@95: {fpr95:.4f} | "
        f"AP: {ap:.4f}"
    )

    return auc, fpr95, ap, train_time


# ============================================================
# AutoEncoder
# ============================================================

def run_ae(
    X_inlier,
    X_anomaly,
    X_test,
    y_test,
):

    logger.info("Running AE...")

    ae_args = {
        "contamination": 0.1,
        "hidden_neuron_list": [64, 32, 16],
        "hidden_activation_name": "relu",
        "epoch_num": 30,
        "batch_size": X_inlier.shape[0] // 5,
        "dropout_rate": 0.0,
        "verbose": 0,
    }

    model = AE(ae_args)

    start = time.time()

    X_train = torch.concatenate(
        [
            X_inlier,
            X_anomaly,
        ]
    )

    model.train(
        X_train.cpu()
    )

    train_time = time.time() - start

    auc, fpr95, ap = model.test(
        X_test,
        y_test,
    )

    logger.info(
        f"AE --> "
        f"AUC: {auc:.4f} | "
        f"FPR@95: {fpr95:.4f} | "
        f"AP: {ap:.4f}"
    )

    return auc, fpr95, ap, train_time


# ============================================================
# TCCM
# ============================================================

def run_tccm(
    X_inlier,
    X_anomaly,
    X_test,
    y_test,
    device,
):

    logger.info("Running TCCM...")

    tccm_args = {
        "n_features": X_inlier.shape[1],
        "epochs": 50,
        "learning_rate": 1e-3,
        "batch_size": 32,
        "device": device,
    }

    model = TCCM(tccm_args)

    start = time.time()

    X_train = torch.concatenate(
        [
            X_inlier,
            X_anomaly,
        ]
    )

    model.train(X_train)

    train_time = time.time() - start

    auc, fpr95, ap = model.test(
        X_test,
        y_test,
    )

    logger.info(
        f"TCCM --> "
        f"AUC: {auc:.4f} | "
        f"FPR@95: {fpr95:.4f} | "
        f"AP: {ap:.4f}"
    )

    return auc, fpr95, ap, train_time


# ============================================================
# Main
# ============================================================

def main():

    args = parse_args()

    device = get_device(args.device)

    logger.info("=" * 70)
    logger.info("TEXTUAL ANOMALY DETECTION - EXPERIMENTS")
    logger.info("=" * 70)

    logger.info(f"Dataset      : {args.dataset}")
    logger.info(f"Inlier topic : {args.inlier_topic}")
    logger.info(f"TAC type     : {args.type_tac}")
    logger.info(f"Nu           : {args.nu}")
    logger.info(f"Model        : {args.model_name}")
    logger.info(f"Device       : {device}")
    logger.info(f"Runs         : {args.nb_runs}")

    # ========================================================
    # Dataset preparation
    # ========================================================

    logger.info("\nPreparing datasets...")

    (
        train_inlier,
        train_anomaly,
        test_inliers,
        test_anomalies,
    ) = get_datasets()

    logger.info("Datasets ready.")

    logger.info(
        f"Train inlier  : {len(train_inlier)}"
    )

    logger.info(
        f"Train anomaly : {len(train_anomaly)}"
    )

    logger.info(
        f"Number of test runs : {len(test_inliers)}"
    )

    # ========================================================
    # Convert embeddings to tensors
    # ========================================================

    # TODO:
    # Adapter ces deux lignes à la manière dont tes datasets
    # stockent réellement les embeddings.

    X_inlier = torch.tensor(
        np.array(train_inlier[args.embedding_name])
    ).to(device)

    X_anomaly = torch.tensor(
        np.array(train_anomaly[args.embedding_name])
    ).to(device)

    logger.info(
        f"X_inlier shape  : {X_inlier.shape}"
    )

    logger.info(
        f"X_anomaly shape : {X_anomaly.shape}"
    )

    # ========================================================
    # Results
    # ========================================================

    results = initialize_results(args)

    # ========================================================
    # Test runs
    # ========================================================

    for n_run in range(args.nb_runs):

        logger.info("\n" + "=" * 70)
        logger.info(
            f"RUN {n_run + 1}/{args.nb_runs}"
        )
        logger.info("=" * 70)

        test_inlier = test_inliers[n_run]
        test_anomaly = test_anomalies[n_run]

        # ----------------------------------------------------
        # Build test dataset
        # ----------------------------------------------------

        data_test, y_test = prepare_test_data(
            test_inlier,
            test_anomaly,
        )

        # ----------------------------------------------------
        # Test embeddings
        # ----------------------------------------------------

        X_test_inlier = torch.tensor(
            np.array(test_inlier[args.embedding_name])
        ).to(device)

        X_test_anomaly = torch.tensor(
            np.array(test_anomaly[args.embedding_name])
        ).to(device)

        X_test = torch.concatenate(
            [
                X_test_inlier,
                X_test_anomaly,
            ]
        )

        logger.info(
            f"X_test shape : {X_test.shape}"
        )

        # ====================================================
        # OCSVM
        # ====================================================

        if args.ocsvm:

            auc, fpr, ap, train_time = run_ocsvm(
                X_inlier,
                X_anomaly,
                X_test,
                y_test,
                device,
            )

            results["ocsvm"]["auc"].append(auc)
            results["ocsvm"]["fpr"].append(fpr)
            results["ocsvm"]["ap"].append(ap)
            results["ocsvm"]["time"].append(train_time)

        # ====================================================
        # RSRAE
        # ====================================================

        if args.rsrae:

            auc, fpr, ap, train_time = run_rsrae(
                X_inlier.mean(dim=1),
                X_anomaly.mean(dim=1),
                X_test.mean(dim=1),
                y_test,
                device,
            )

            results["rsrae"]["auc"].append(auc)
            results["rsrae"]["fpr"].append(fpr)
            results["rsrae"]["ap"].append(ap)
            results["rsrae"]["time"].append(train_time)

        # ====================================================
        # AE
        # ====================================================

        if args.ae:

            auc, fpr, ap, train_time = run_ae(
                X_inlier.mean(dim=1),
                X_anomaly.mean(dim=1),
                X_test.mean(dim=1),
                y_test,
            )

            results["ae"]["auc"].append(auc)
            results["ae"]["fpr"].append(fpr)
            results["ae"]["ap"].append(ap)
            results["ae"]["time"].append(train_time)

        # ====================================================
        # TCCM
        # ====================================================

        if args.tccm:

            auc, fpr, ap, train_time = run_tccm(
                X_inlier.mean(dim=1),
                X_anomaly.mean(dim=1),
                X_test.mean(dim=1),
                y_test,
                device,
            )

            results["tccm"]["auc"].append(auc)
            results["tccm"]["fpr"].append(fpr)
            results["tccm"]["ap"].append(ap)
            results["tccm"]["time"].append(train_time)

    # ========================================================
    # Final results
    # ========================================================

    logger.info("\n" + "=" * 70)
    logger.info("FINAL RESULTS")
    logger.info("=" * 70)

    for model_name, model_results in results.items():

        if len(model_results["auc"]) == 0:
            continue

        auc_mean = np.mean(model_results["auc"])
        auc_std = np.std(model_results["auc"])

        fpr_mean = np.mean(model_results["fpr"])
        fpr_std = np.std(model_results["fpr"])

        ap_mean = np.mean(model_results["ap"])
        ap_std = np.std(model_results["ap"])

        time_mean = np.mean(model_results["time"])

        logger.info(
            f"\n{model_name.upper()}"
        )

        logger.info(
            f"AUC      : {auc_mean:.4f} ± {auc_std:.4f}"
        )

        logger.info(
            f"FPR@95   : {fpr_mean:.4f} ± {fpr_std:.4f}"
        )

        logger.info(
            f"AP       : {ap_mean:.4f} ± {ap_std:.4f}"
        )

        logger.info(
            f"Train time : {time_mean:.2f} sec"
        )

        # ====================================================
        # TODO: save_results
        # ====================================================

        # save_results(
        #     dataset_name=args.dataset,
        #     inlier_topic=args.inlier_topic,
        #     type_emb=args.model_name,
        #     ad_model=model_name,
        #     auc_mean=auc_mean,
        #     ap_mean=ap_mean,
        #     fpr_mean=fpr_mean,
        #     auc_std=auc_std,
        #     ap_std=ap_std,
        #     fpr_std=fpr_std,
        #     train_time=time_mean,
        #     nu=args.nu,
        #     overwrite="naive",
        #     )

    logger.info("\nDone.")


if __name__ == "__main__":
    main()