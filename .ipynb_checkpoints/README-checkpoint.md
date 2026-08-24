# FLOCAT: FLOw Matching toward a CompAct Target Distribution for Text Anomaly Detection

> Paper submitted to EMNLP 2026.

## Overview

FLOCAT is an unsupervised framework for text anomaly detection based on
source-reversed flow matching. Rather than learning to generate text, FLOCAT
learns a continuous transport from inlier text embeddings toward a compact
isotropic Gaussian target. The distance to the target centroid serves as a
geometrically consistent anomaly score.

Built on a Diffusion Transformer (DiT) backbone, FLOCAT handles both
sentence-level and token-level inputs in a single forward pass, and produces
token-level attribution scores as a zero-overhead by-product of inference.

## Key Features

- Source-reversed flow matching with a low-variance Gaussian target and LOVE constraint
- Level-agnostic DiT architecture with residual injection and velocity pooling
- Token-level attribution scores $\tilde{a}^{i,k}$ without additional supervision
- Evaluated on 10 datasets across 5 encoder backbones

## Installation

```bash
git clone https://github.com/[username]/flocat
cd flocat
pip install -r requirements.txt
```

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

```bibtex
@article{flocat2026,
  title={FLOCAT: FLOw Matching toward a CompAct Target Distribution
         for Text Anomaly Detection},
  author={...},
  journal={EMNLP},
  year={2026}
}
```

## License

MIT License