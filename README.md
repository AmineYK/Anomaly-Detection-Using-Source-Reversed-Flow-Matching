> Paper published to EMNLP 2026.

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
@inproceedings{youcef-khodja-etal-2026-anomaly,
  title = "Anomaly Detection using Source-Reversed Flow Matching",
  author = {Youcef Khodja, Amine and
    Pantin, Jeremie and
    Dias, Ga{\"e}l and
    Maurel, Fabrice},
  booktitle = "Proceedings of the 2026 Conference on Empirical Methods in Natural Language Processing",
  year = "2026",
  publisher = "Association for Computational Linguistics"
}
```