> Paper published at EMNLP 2026.

## Overview

The paper ***Anomaly Detection using Source-Reversed Flow Matching*** reverses the usual direction of Flow Matching. Instead of generating text, we learn to transport representations of normal documents toward a compact Gaussian distribution. The distance to this target then provides a geometrically grounded anomaly score.
The method also provides token-level explanations without requiring any additional supervision. We achieve state-of-the-art performance across 10 datasets and 5 different encoders, with applications ranging from machine-generated text detection to depression screening.

## Key Features

- Source-reversed flow matching with a low-variance Gaussian target and LOVE constraint
- Level-agnostic DiT architecture with residual injection and velocity pooling
- Token-level attribution scores without additional supervision
- Evaluated on 10 datasets across 5 encoder backbones


## Usage

### Sentence-level

To run the experiments at the sentence level:

```bash
python3 run_sentence_level.py \
    --dataset_name "reuters" \
    --inlier_topic "acq" \
    --type_tac "ruff" \
    --nu 0.1 \
    --model_name "all-mpnet-base-v2" \
    --embedding_name "mpnet_embedding" \
    --nb_runs 5 \
    --flocat --rsrae --tccm --fate --llm --llm_name "..."
```

### Token-level

To run the experiments at the token level:

```bash
python3 main_token_level.py \
    --dataset_name "reuters" \
    --inlier_topic "acq" \
    --type_tac "ruff" \
    --nu 0.1 \
    --nb_runs 5 \
    --seq_len 128 \
    --type_emb "roberta" \
    --flocat --rsrae --tccm --date --cvdd
```

## Datasets

We evaluate on 10 datasets: 20Newsgroups, Reuters, AGNews, DBpedia14,
SMSSpam, Enron, IMDB, SST-2, M4, and DAIC-WoZ.

## Results

State-of-the-art AUC-ROC across all main benchmarks and encoder backbones.
See the paper for full results.

## Citation

```bibtex 
@inproceedings{khodja-etal-2026-anomaly,
    title = "Anomaly Detection using Source-Reversed Flow Matching",
    author = {Khodja, Amine Youcef  and
      Pantin, Jeremie  and
      Dias, Ga{\"e}l  and
      Maurel, Fabrice},
    editor = "Lee, Hung-Yi  and
      Dev, Sunipa  and
      Li, Jessy  and
      Elliott, Desmond",
    booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
    month = oct,
    year = "2026",
    address = "Budapest, Hungary",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.emnlp-main.2322/",
    doi = "10.18653/2026.emnlp-main.2322",
    pages = "52746--52767",
    ISBN = "979-8-89176-459-0",
    abstract = "Anomaly detection in textual data remains an underexplored problem, hindered by fragmented benchmarks, scarce labeled data, and methods that fail to model the distribution of normal text in a continuous and generative manner. We propose Flocat, an unsupervised approach that reformulates this task through a source-reversed flow matching paradigm: rather than learning to generate, we learn a flow from inlier text embeddings toward a compact isotropic Gaussian target, whose radius is explicitly constrained to ensure that the distance to the centroid serves as a geometrically consistent anomaly score. Built on a Diffusion Transformer (DiT) backbone, Flocat operates at both token and sentence level through a level-agnostic interpolation mechanism, enabling fine-grained signal interpretability. We evaluate on 10 datasets spanning text classification, spam detection, sentiment analysis, human- vs. machine-generated text and depression screening, consistently achieving state-of-the-art AUC-ROC across all benchmarks and text representations."
}
```
