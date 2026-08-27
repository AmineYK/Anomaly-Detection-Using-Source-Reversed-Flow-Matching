"""
Chargement, préprocessing et contamination (TAC) des datasets.
Remplace le chargement local par le pipeline utils/tac fourni.
Ce module est importé par run_sentence_level.py et run_token_level.py.
"""

import logging
import torch

import Data.utils as utils
import Data.tac as tac

logger = logging.getLogger(__name__)

DATASET_TOPICS_DICT = {
    "20newsgroups": ["computer", "recreation", "science", "miscellaneous", "politics", "religion"],
    "reuters": ["earn", "trade", "acq", "money-fx", "crude", "ship", "interest"],
    "agnews": ["World", "Sports", "Business", "Sci-Tech"],
    "dbpedia14": [
        "Company", "Educational Institution", "Artist", "Athlete", "Office Holder",
        "Mean Of Transportation", "Building", "Natural Place", "Village", "Animal",
        "Plant", "Album", "Film", "Written Work",
    ],
    "sms": ["normal"],
    "imdb": ["positive", "negative"],
    "sst2": ["positive", "negative"],
    "enron": ["normal"],
    "m4": ["wikipedia", "arxiv", "wikihow", "reddit", "peerread"],
}

DATASET_COL = {"dbpedia14": "content"}
DEFAULT_COL = "text"


def get_text_column(dataset_name):
    return DATASET_COL.get(dataset_name, DEFAULT_COL)


def get_device(device_arg):
    if device_arg is not None:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_inlier_topics(dataset_name, inlier_topic, runall):
    available_topics = DATASET_TOPICS_DICT[dataset_name]

    if not available_topics:
        raise ValueError(f"No inlier topics defined for dataset '{dataset_name}'.")

    if runall:
        return available_topics

    if inlier_topic not in available_topics:
        raise ValueError(
            f"Invalid inlier topic '{inlier_topic}' for dataset '{dataset_name}'.\n"
            f"Available topics:\n{available_topics}"
        )

    return [inlier_topic]


def load_dataset(dataset_name, batch_size):
    logger.info(f"Loading dataset: {dataset_name}")

    train_dataset, test_dataset = utils.import_dataset(
        name=dataset_name,
        batch_size=batch_size,
    )

    return train_dataset.dataset, test_dataset.dataset


def preprocess_dataset(dataset_name, train_dataset, test_dataset):
    logger.info(f"Preprocessing {dataset_name}...")

    if dataset_name in ("sms", "imdb", "sst2"):
        train_dataset = utils.unify_text_column(train_dataset, dataset_name)
        test_dataset = utils.unify_text_column(test_dataset, dataset_name)
        train_dataset = utils.preprocess(train_dataset)
        test_dataset = utils.preprocess(test_dataset)

    elif dataset_name == "dbpedia14":
        train_dataset = utils.preprocess(train_dataset, "content")
        test_dataset = utils.preprocess(test_dataset, "content")

    else:
        train_dataset = utils.preprocess(train_dataset)
        test_dataset = utils.preprocess(test_dataset)

    return train_dataset, test_dataset


def apply_tac(dataset, dataset_name, inlier_topic, type_tac, nu):
    """Textual Anomaly Contamination : renvoie (inlier_split, anomaly_split)."""
    logger.info(
        f"Applying TAC | dataset={dataset_name} | topic={inlier_topic} "
        f"| type_tac={type_tac} | nu={nu}"
    )
    return tac.textual_anomaly_contamination(
        dataset,
        dataset_name,
        inlier_topic,
        type_tac,
        nu,
        True,
    )


def build_labeled_test_set(test_inlier, test_anomaly):
    """
    Concatène inlier/anomaly de test et fabrique le vecteur de labels y_test
    (0 = inlier, 1 = anomalie). Remplace la colonne 'anomaly_class' qui était
    déjà présente dans les anciens fichiers pré-construits.
    """
    from datasets import concatenate_datasets
    import numpy as np

    test_dataset = concatenate_datasets([test_inlier, test_anomaly])
    y_test = np.array([0] * len(test_inlier) + [1] * len(test_anomaly))

    return test_dataset, y_test