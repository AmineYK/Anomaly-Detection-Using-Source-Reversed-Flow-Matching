# Anomaly Detection using Source-Reversed Flow Matching

> Paper submitted to EMNLP 2026.

## Overview

The paper “Anomaly Detection using Source-Reversed Flow Matching” reverses the usual direction of Flow Matching. Instead of generating text, we learn to transport representations of normal documents toward a compact Gaussian distribution. The distance to this target then provides a geometrically grounded anomaly score.
The method also provides token-level explanations without requiring any additional supervision. We achieve state-of-the-art performance across 10 datasets and 5 different encoders, with applications ranging from machine-generated text detection to depression screening.

## Key Features

- Source-reversed flow matching with a low-variance Gaussian target and LOVE constraint
- Level-agnostic DiT architecture with residual injection and velocity pooling
- Token-level attribution scores without additional supervision
- Evaluated on 10 datasets across 5 encoder backbones


## Usage

```bash
# Training
python scripts/train.py --config configs/datasets/reuters.yaml --encoder roberta

# Evaluation
python scripts/evaluate.py --checkpoint path/to/checkpoint

# Attribution faithfulness
python scripts/faithfulness.py --dataset m4 --k 10 20
```

## Datasets

We evaluate on 10 datasets: 20Newsgroups, Reuters, AGNews, DBpedia14,
SMSSpam, Enron, IMDB, SST-2, M4, and DAIC-WoZ.
See `data/README.md` for download instructions.

## Results

State-of-the-art AUC-ROC across all main benchmarks and encoder backbones.
See the paper for full results.

## Citation

```bibtex XXXXXXXXX
```

## License

XXXXXX