from datasets import concatenate_datasets
import numpy as np

def _add_anomaly_column(dataset, value):
    return dataset.map(
        lambda x: {"anomaly_class": value},
        load_from_cache_file=False,
        num_proc=1,
    )


def _compute_n_anomalies(n_inliers, anomaly_rate):
    return int((anomaly_rate * n_inliers) / (1 - anomaly_rate))


def _sample_dataset(dataset, n_samples):
    indices = np.random.randint(0, dataset.num_rows, n_samples)
    return dataset.select(indices)


def _train_split(inlier_dataset, anomaly_dataset, anomaly_rate):
    n_anomalies = _compute_n_anomalies(
        inlier_dataset.num_rows,
        anomaly_rate,
    )

    anomaly_dataset = _sample_dataset(anomaly_dataset, n_anomalies)

    inlier_dataset = _add_anomaly_column(inlier_dataset, 0)
    anomaly_dataset = _add_anomaly_column(anomaly_dataset, 1)

    return inlier_dataset, anomaly_dataset


def _test_split(inlier_dataset, anomaly_dataset, anomaly_rate):
    n_inliers = inlier_dataset.num_rows
    n_replace = int(n_inliers * anomaly_rate)

    keep_indices = np.random.choice(
        n_inliers,
        n_inliers - n_replace,
        replace=False,
    )
    anomaly_indices = np.random.choice(
        anomaly_dataset.num_rows,
        n_replace,
        replace=False,
    )

    inlier_dataset = inlier_dataset.select(keep_indices)
    anomaly_dataset = anomaly_dataset.select(anomaly_indices)

    inlier_dataset = _add_anomaly_column(inlier_dataset, 0)
    anomaly_dataset = _add_anomaly_column(anomaly_dataset, 1)

    return concatenate_datasets(
        [inlier_dataset, anomaly_dataset]
    ).shuffle(seed=42)


def _finalize_split(
    inlier_dataset,
    anomaly_dataset,
    anomaly_rate,
    is_trainset,
):
    if is_trainset:
        return _train_split(
            inlier_dataset,
            anomaly_dataset,
            anomaly_rate,
        )

    return _test_split(
        inlier_dataset,
        anomaly_dataset,
        anomaly_rate,
    )


def textual_anomaly_contamination(
    dataset,
    dataset_name,
    inlier_topic,
    type_tac,
    anomaly_rate=0.1,
    is_trainset=True,
):
    handlers = {
        "reuters": textual_anomaly_contamination_reuters,
        "wos": textual_anomaly_contamination_wos,
        "dbpedia14": textual_anomaly_contamination_dbpedia14,
        "agnews": textual_anomaly_contamination_agnews,
        "20newsgroups": textual_anomaly_contamination_20newsgroups,
        "sms": textual_anomaly_contamination_binary,
        "enron": textual_anomaly_contamination_binary,
        "imdb": textual_anomaly_contamination_binary,
        "sst2": textual_anomaly_contamination_binary,
        "mage": textual_anomaly_contamination_binary,
        "m4": textual_anomaly_contamination_m4,
    }

    if dataset_name not in handlers:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return handlers[dataset_name](
        dataset,
        dataset_name,
        inlier_topic,
        type_tac,
        anomaly_rate,
        is_trainset,
    )


def textual_anomaly_contamination_binary(
    dataset,
    dataset_name,
    inlier_topic,
    type_tac=None,
    anomaly_rate=0.1,
    is_trainset=True,
):
    """Create inlier and anomaly splits for binary datasets."""

    if dataset_name in ("sst2", "imdb"):
        if inlier_topic == "positive":
            inlier_label = 1
            anomaly_label = 0
        elif inlier_topic == "negative":
            inlier_label = 0
            anomaly_label = 1
    else:
        if inlier_topic == "normal":
            inlier_label = 0
            anomaly_label = 1
        else:
            inlier_label = 1
            anomaly_label = 0

    inlier_dataset = dataset.filter(
        lambda x: x["label"] == inlier_label
    )
    anomaly_dataset = dataset.filter(
        lambda x: x["label"] == anomaly_label
    )

    return _finalize_split(
        inlier_dataset,
        anomaly_dataset,
        anomaly_rate,
        is_trainset,
    )


def textual_anomaly_contamination_dbpedia14(
    dataset,
    dataset_name,
    inlier_topic,
    type_tac="pantin",
    anomaly_rate=0.1,
    is_trainset=True,
):
    if type_tac != "pantin":
        raise ValueError("TAC not available")

    level_1_mapping = {
        "Company": 0,
        "Educational Institution": 1,
        "Artist": 2,
        "Athlete": 3,
        "Office Holder": 4,
        "Mean Of Transportation": 5,
        "Building": 6,
        "Natural Place": 7,
        "Village": 8,
        "Animal": 9,
        "Plant": 10,
        "Album": 11,
        "Film": 12,
        "Written Work": 13,
    }

    if inlier_topic not in level_1_mapping:
        raise ValueError(f"Invalid inlier_topic: {inlier_topic}")

    inlier_label = level_1_mapping[inlier_topic]

    inlier_indices = []
    anomaly_indices = []

    for i, label in enumerate(dataset["label"]):
        if label == inlier_label:
            inlier_indices.append(i)
        else:
            anomaly_indices.append(i)

    inlier_dataset = dataset.select(inlier_indices)
    anomaly_dataset = dataset.select(anomaly_indices)

    return _finalize_split(
        inlier_dataset,
        anomaly_dataset,
        anomaly_rate,
        is_trainset,
    )


def textual_anomaly_contamination_m4(
    dataset,
    dataset_name,
    inlier_topic,
    type_tac=None,
    anomaly_rate=0.1,
    is_trainset=True,
):
    """Create machine-generated inliers and human-generated anomalies."""

    inlier_indices = []
    anomaly_indices = []

    for i, sample in enumerate(dataset):
        domain = sample["inlier_topic"]
        label = sample["label"]

        if label == 0 and domain == inlier_topic:
            inlier_indices.append(i)
        elif label == 1 and domain == inlier_topic:
            anomaly_indices.append(i)

    inlier_dataset = dataset.select(inlier_indices)
    anomaly_dataset = dataset.select(anomaly_indices)

    return _finalize_split(
        inlier_dataset,
        anomaly_dataset,
        anomaly_rate,
        is_trainset,
    )


def textual_anomaly_contamination_20newsgroups(
    dataset,
    dataset_name,
    inlier_topic,
    type_tac="ruff",
    anomaly_rate=0.1,
    is_trainset=True,
):
    if type_tac == "ruff":
        groups = {
            "computer": [
                "comp.graphics",
                "comp.os.ms-windows.misc",
                "comp.sys.ibm.pc.hardware",
                "comp.sys.mac.hardware",
                "comp.windows.x",
            ],
            "recreation": [
                "rec.autos",
                "rec.motorcycles",
                "rec.sport.baseball",
                "rec.sport.hockey",
            ],
            "science": [
                "sci.crypt",
                "sci.electronics",
                "sci.med",
                "sci.space",
            ],
            "miscellaneous": ["misc.forsale"],
            "politics": [
                "talk.politics.misc",
                "talk.politics.guns",
                "talk.politics.mideast",
            ],
            "religion": [
                "talk.religion.misc",
                "alt.atheism",
                "soc.religion.christian",
            ],
        }

    elif type_tac == "pantin":
        groups = {
            "computer": [
                "comp.graphics",
                "comp.os.ms-windows.misc",
                "comp.sys.ibm.pc.hardware",
                "comp.sys.mac.hardware",
                "comp.windows.x",
            ],
            "motors": [
                "rec.motorcycles",
                "rec.autos",
            ],
            "sports": [
                "rec.sport.baseball",
                "rec.sport.hockey",
            ],
            "science": [
                "sci.crypt",
                "sci.electronics",
                "sci.med",
                "sci.space",
            ],
            "forsale": ["misc.forsale"],
            "politics": [
                "talk.politics.misc",
                "talk.politics.guns",
                "talk.politics.mideast",
            ],
            "religion": [
                "talk.religion.misc",
                "alt.atheism",
                "soc.religion.christian",
            ],
        }

    else:
        raise ValueError("TAC not available.")

    topic_map = {
        label: group
        for group, labels in groups.items()
        for label in labels
    }

    dataset = dataset.map(
        lambda x: {
            "topic_label_text": topic_map.get(
                x["label_text"],
                "unknown",
            )
        }
    )

    if inlier_topic in groups:
        inlier_subtopics = groups[inlier_topic]
        anomaly_subtopics = [
            label
            for group_name, labels in groups.items()
            if group_name != inlier_topic
            for label in labels
        ]
    else:
        group_found = None

        for group_name, labels in groups.items():
            if inlier_topic in labels:
                group_found = group_name
                break

        if group_found is None:
            raise ValueError(
                f"{inlier_topic} does not exist in any group."
            )

        inlier_subtopics = [inlier_topic]
        anomaly_subtopics = [
            label
            for label in groups[group_found]
            if label != inlier_topic
        ]

        if len(anomaly_subtopics) == 0:
            raise ValueError(
                "Cannot create anomalies: group contains only one class."
            )

    inlier_dataset = dataset.filter(
        lambda x: x["label_text"] in inlier_subtopics
    )
    anomaly_dataset = dataset.filter(
        lambda x: x["label_text"] in anomaly_subtopics
    )

    return _finalize_split(
        inlier_dataset,
        anomaly_dataset,
        anomaly_rate,
        is_trainset,
    )


def textual_anomaly_contamination_reuters(
    dataset,
    dataset_name,
    inlier_topic,
    type_tac="ruff",
    anomaly_rate=0.1,
    is_trainset=True,
):
    dataset = dataset.filter(
        lambda x: len(x["topics"]) == 1
    )

    if type_tac == "ruff":
        if is_trainset:
            values, counts = np.unique(
                dataset[:]["topics"],
                return_counts=True,
            )
            selected_labels = values[counts >= 100]
            dataset = dataset.filter(
                lambda x: x["topics"] in selected_labels
            )

            if inlier_topic not in selected_labels:
                raise ValueError("Inlier topic invalid for RUFF")

        inlier_dataset = dataset.filter(
            lambda x: x["topics"] == [inlier_topic]
        )
        anomaly_dataset = dataset.filter(
            lambda x: x["topics"] != [inlier_topic]
        )

        return _finalize_split(
            inlier_dataset,
            anomaly_dataset,
            anomaly_rate,
            is_trainset,
        )

    if type_tac == "pantin":
        parent_topics = {
            "commodities": [
                "acq",
                "carcass",
                "cocoa",
                "coconut",
                "coffee",
                "cotton",
                "grain",
                "groundnut",
                "hog",
                "housing",
                "l-cattle",
                "livestock",
                "lumber",
                "meal-feed",
                "oilseed",
                "orange",
                "potato",
                "rice",
                "rubber",
                "ship",
                "sugar",
                "tea",
                "veg-oil",
            ],
            "financial": [
                "cpi",
                "cpu",
                "dlr",
                "earn",
                "income",
                "instal-debt",
                "interest",
                "ipi",
                "jobs",
                "lei",
                "money-fx",
                "money-supply",
                "rand",
                "reserves",
                "retail",
                "trade",
                "wpi",
                "yen",
            ],
            "metals": [
                "alum",
                "copper",
                "gold",
                "iron-steel",
                "lead",
                "nickel",
                "platinum",
                "silver",
                "strategic-metal",
                "tin",
                "zinc",
            ],
            "energy": [
                "crude",
                "fuel",
                "heat",
                "jet",
                "naphtha",
                "nat-gas",
                "propane",
            ],
        }

        topic_map = {
            topic: parent
            for parent, topics in parent_topics.items()
            for topic in topics
        }

        dataset = dataset.map(
            lambda x: {
                "parent_topic": topic_map.get(
                    x["topics"][0],
                    "unknown",
                )
            }
        )

        if inlier_topic not in parent_topics:
            raise ValueError("Invalid inlier topic")

        inlier_dataset = dataset.filter(
            lambda x: x["parent_topic"] == inlier_topic
        )
        anomaly_dataset = dataset.filter(
            lambda x: x["parent_topic"] != inlier_topic
        )

        return _finalize_split(
            inlier_dataset,
            anomaly_dataset,
            anomaly_rate,
            is_trainset,
        )

    raise ValueError("type_tac not available")


def textual_anomaly_contamination_wos(
    dataset,
    dataset_name,
    inlier_topic,
    type_tac="pantin",
    anomaly_rate=0.1,
    is_trainset=True,
):
    if type_tac != "pantin":
        raise ValueError("TAC not available")

    mapping = {
        "Computer_Science": 0,
        "Electrical_Engineering": 1,
        "Psychology": 2,
        "Mechanical_Engineering": 3,
        "Civil_Engineering": 4,
        "Medical_Science": 5,
        "Biochemistry": 6,
    }

    if inlier_topic not in mapping:
        raise ValueError("Invalid inlier topic")

    label = mapping[inlier_topic]

    inlier_dataset = dataset.filter(
        lambda x: x["label_level_1"] == label
    )
    anomaly_dataset = dataset.filter(
        lambda x: x["label_level_1"] != label
    )

    return _finalize_split(
        inlier_dataset,
        anomaly_dataset,
        anomaly_rate,
        is_trainset,
    )


def textual_anomaly_contamination_agnews(
    dataset,
    dataset_name,
    inlier_topic,
    type_tac="fate",
    anomaly_rate=0.1,
    is_trainset=True,
):
    if type_tac != "fate":
        raise ValueError("TAC not available")

    mapping = {
        "World": 0,
        "Sports": 1,
        "Business": 2,
        "Sci/Tech": 3,
    }

    if inlier_topic not in mapping:
        raise ValueError("Invalid inlier topic")

    label = mapping[inlier_topic]

    inlier_dataset = dataset.filter(
        lambda x: x["label"] == label
    )
    anomaly_dataset = dataset.filter(
        lambda x: x["label"] != label
    )

    return _finalize_split(
        inlier_dataset,
        anomaly_dataset,
        anomaly_rate,
        is_trainset,
    )