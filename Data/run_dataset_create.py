import argparse
import os
import logging
import torch

import utils
import embedding_encoder
import tac

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

DATASET_TOPICS_DICT = {
    "20newsgroups": [
        "computer",
        "recreation",
        "science",
        "miscellaneous",
        "politics",
        "religion",
    ],

    "reuters": [
        "earn",
        "trade",
        "acq",
        "money-fx",
        "crude",
        "ship",
        "interest",
    ],

    "agnews": [
        "World",
        "Sports",
        "Business",
        "Sci-Tech",
    ],

    "dbpedia14": [
        "Company",
        "Educational Institution",
        "Artist",
        "Athlete",
        "Office Holder",
        "Mean Of Transportation",
        "Building",
        "Natural Place",
        "Village",
        "Animal",
        "Plant",
        "Album",
        "Film",
        "Written Work",
    ],

    "sms": [
        "normal",
    ],

    "imdb": [
        "positive",
        "negative",
    ],

    "sst2": [
        "positive",
        "negative",
    ],

    "enron": [
        "normal",
    ],

    "m4": [
        "wikipedia",
        "arxiv",
        "wikihow",
        "reddit",
        "peerread",
    ],
}


def parse_args():

    parser = argparse.ArgumentParser(
        description="Textual Anomaly Detection Framework"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_TOPICS_DICT.keys()),
        help="Dataset to process.",
    )

    parser.add_argument(
        "--inlier_topic",
        type=str,
        default=None,
        help=(
            "Inlier topic. If not specified, "
            "all available topics are processed."
        ),
    )

    parser.add_argument(
        "--type_tac",
        type=str,
        default="pantin",
        choices=[
            "ruff",
            "pantin",
            "fate",
        ],
        help="Type of Textual Anomaly Contamination.",
    )

    parser.add_argument(
        "--nu",
        type=float,
        default=0.1,
        help="Anomaly contamination rate.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="all-mpnet-base-v2",
        help="SentenceTransformer model name.",
    )

    parser.add_argument(
        "--embedding_name",
        type=str,
        default="mpnet_embedding",
        help="Name of the embedding column.",
    )

    parser.add_argument(
        "--encoder_type",
        type=str,
        default="sentencebert",
        help="Embedding encoder type.",
    )

    parser.add_argument(
        "--nb_runs",
        type=int,
        default=5,
        help="Number of Runs.",
    )

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
        help="Device (cpu, cuda, cuda:0, ...).",
    )

    return parser.parse_args()


def get_device(device_arg):

    if device_arg is not None:
        return torch.device(device_arg)

    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


def load_dataset(dataset_name, batch_size):

    logger.info(f"Loading dataset: {dataset_name}")

    train_dataset, test_dataset = utils.import_dataset(
        name=dataset_name,
        batch_size=batch_size,
    )

    train_dataset = train_dataset.dataset
    test_dataset = test_dataset.dataset

    return train_dataset, test_dataset


def preprocess_dataset(
    dataset_name,
    train_dataset,
    test_dataset,
):

    logger.info(f"Preprocessing {dataset_name}...")

    if dataset_name == "sms":

        train_dataset = utils.unify_text_column(
            train_dataset,
            "sms",
        )

        test_dataset = utils.unify_text_column(
            test_dataset,
            "sms",
        )

        train_dataset = utils.preprocess(train_dataset)
        test_dataset = utils.preprocess(test_dataset)

    elif dataset_name == "imdb":

        train_dataset = utils.unify_text_column(
            train_dataset,
            "imdb",
        )

        test_dataset = utils.unify_text_column(
            test_dataset,
            "imdb",
        )

        train_dataset = utils.preprocess(train_dataset)
        test_dataset = utils.preprocess(test_dataset)

    elif dataset_name == "sst2":

        train_dataset = utils.unify_text_column(
            train_dataset,
            "sst2",
        )

        test_dataset = utils.unify_text_column(
            test_dataset,
            "sst2",
        )

        train_dataset = utils.preprocess(train_dataset)
        test_dataset = utils.preprocess(test_dataset)

    elif dataset_name == "dbpedia14":

        train_dataset = utils.preprocess(
            train_dataset,
            "content",
        )

        test_dataset = utils.preprocess(
            test_dataset,
            "content",
        )

    else:

        train_dataset = utils.preprocess(
            train_dataset,
        )

        test_dataset = utils.preprocess(
            test_dataset,
        )

    return train_dataset, test_dataset


# ============================================================
# Textual Anomaly Contamination
# ============================================================

def apply_tac(
    train_dataset,
    dataset_name,
    inlier_topic,
    type_tac,
    nu,
):

    logger.info(
        f"\nApplying TAC:"
        f"\n  Dataset      : {dataset_name}"
        f"\n  Inlier topic : {inlier_topic}"
        f"\n  TAC type     : {type_tac}"
        f"\n  Nu           : {nu}"
    )

    train_inlier, train_anomaly = (
        tac.textual_anomaly_contamination(
            train_dataset,
            dataset_name,
            inlier_topic,
            type_tac,
            nu,
            True,
        )
    )

    return train_inlier, train_anomaly


# ============================================================
# Embedding
# ============================================================

def create_encoder(
    model_name,
    embedding_name,
    encoder_type,
    device,
):

    logger.info(
        f"\nCreating embedding encoder:"
        f"\n  Model     : {model_name}"
        f"\n  Embedding : {embedding_name}"
        f"\n  Type      : {encoder_type}"
        f"\n  Device    : {device}"
    )

    encoder = embedding_encoder.EmbeddingEncoder(
        model_name,
        embedding_name,
        encoder_type,
        device,
    )

    return encoder


def embed_dataset(
    dataset,
    encoder,
    text_column,
):

    return encoder.forward(
        dataset,
        text_column,
    )


def get_datasets():

    logger.info("Let's do it !!")

    args = parse_args()
    device = get_device(args.device)

    logger.info("=" * 70)
    logger.info("TEXTUAL ANOMALY DETECTION")
    logger.info("=" * 70)

    logger.info(f"Dataset       : {args.dataset}")
    logger.info(f"Inlier topic  : {args.inlier_topic}")
    logger.info(f"TAC type      : {args.type_tac}")
    logger.info(f"Nu            : {args.nu}")
    logger.info(f"Embedding     : {args.model_name}")
    logger.info(f"Device        : {device}")
    logger.info(f"Batch size    : {args.batch_size}")

    if not 0 < args.nu <= 1:
        raise ValueError(
            "--nu must be between 0 and 1."
        )

    available_topics = DATASET_TOPICS_DICT[args.dataset]

    if not available_topics:
        raise ValueError(
            f"No inlier topics defined for dataset "
            f"'{args.dataset}'."
        )

    if args.inlier_topic is not None:

        if args.inlier_topic not in available_topics:

            raise ValueError(
                f"Invalid inlier topic "
                f"'{args.inlier_topic}' for "
                f"dataset '{args.dataset}'.\n"
                f"Available topics:\n"
                f"{available_topics}"
            )

        inlier_topics = [args.inlier_topic]

    else:

        inlier_topics = available_topics

    train_dataset, test_dataset = load_dataset(
        args.dataset,
        args.batch_size,
    )

    train_dataset, test_dataset = preprocess_dataset(
        args.dataset,
        train_dataset,
        test_dataset,
    )

    logger.info(
        "\nDataset successfully loaded and preprocessed."
    )

    logger.info(f"Train size: {len(train_dataset)}")
    logger.info(f"Test size : {len(test_dataset)}")

    if args.dataset == "dbpedia14":
        text_column = "content"
    else:
        text_column = "text"

    sentencebert_encoder = create_encoder(
        model_name=args.model_name,
        embedding_name=args.embedding_name,
        encoder_type=args.encoder_type,
        device=device,
    )

    for inlier_topic in inlier_topics:

        logger.info("\n")
        logger.info("=" * 70)
        logger.info(
            f"PROCESSING INLIER TOPIC: {inlier_topic}"
        )
        logger.info("=" * 70)

        logger.info("\nTAC algorithm...")

        train_inlier, train_anomaly = apply_tac(
            train_dataset=train_dataset,
            dataset_name=args.dataset,
            inlier_topic=inlier_topic,
            type_tac=args.type_tac,
            nu=args.nu,
        )

        logger.info("\nEmbedding inlier dataset...")

        train_inlier = embed_dataset(
            dataset=train_inlier,
            encoder=sentencebert_encoder,
            text_column=text_column,
        )

        logger.info("Embedding anomaly dataset...")

        train_anomaly = embed_dataset(
            dataset=train_anomaly,
            encoder=sentencebert_encoder,
            text_column=text_column,
        )

        test_inliers = []
        test_anomalies = []

        for n_run in range(1, args.nb_runs + 1):

            logger.info("\n" + "-" * 70)
            logger.info(
                f"TEST RUN {n_run}/{args.nb_runs}"
            )
            logger.info("-" * 70)

            test_inlier, test_anomaly = apply_tac(
                train_dataset=test_dataset,
                dataset_name=args.dataset,
                inlier_topic=inlier_topic,
                type_tac=args.type_tac,
                nu=args.nu,
            )

            logger.info(
                "Embedding test inlier dataset..."
            )

            test_inlier = embed_dataset(
                dataset=test_inlier,
                encoder=sentencebert_encoder,
                text_column=text_column,
            )

            logger.info(
                "Embedding test anomaly dataset..."
            )

            test_anomaly = embed_dataset(
                dataset=test_anomaly,
                encoder=sentencebert_encoder,
                text_column=text_column,
            )

            test_inliers.append(test_inlier)
            test_anomalies.append(test_anomaly)

    logger.info(
        f"\nFinished processing '{inlier_topic}'."
    )

    return (
        train_inlier,
        train_anomaly,
        test_inliers,
        test_anomalies,
    )


if __name__ == "__main__":

    train_inlier, train_anomaly, test_inliers, test_anomalies = get_datasets()


# python3 run_dataset_create.py \
#     --dataset "reuters" \
#     --inlier_topic "acq" \
#     --type_tac "ruff" \
#     --nu 0.1 \
#     --model_name "all-mpnet-base-v2" \
#     --embedding_name "mpnet_embedding" \
#     --nb_runs 5