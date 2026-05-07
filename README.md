# Earnings Call Sentiment Analysis for Stock Return Prediction

Predicting post-earnings stock movement using NLP on earnings call transcripts.  
Fine-tuned FinBERT + hierarchical transformer to handle long transcripts.

---

## Team

| Person | Role |
|--------|------|
| TBD | Data pipeline |
| TBD | Preprocessing & dataset |
| TBD | FinBERT fine-tuning |
| TBD | Hierarchical transformer |
| TBD | Training loop, eval, demo |

---

## Project Structure

```
earnings-sentiment/
├── data/
│   ├── raw/                  # Raw transcripts + price data (gitignored)
│   └── processed/            # Tokenized, chunked, labeled tensors (gitignored)
├── src/
│   ├── data/
│   │   ├── fetch_transcripts.py   # Person 1: FinancialModelingPrep scraper
│   │   ├── fetch_returns.py       # Person 1: Yahoo Finance return calculator
│   │   ├── labels.py              # Person 1: Label construction (post-open window)
│   │   ├── chunker.py             # Person 2: Transcript chunking
│   │   └── dataset.py             # Person 2: PyTorch Dataset class
│   ├── models/
│   │   ├── finbert_head.py        # Person 3: FinBERT + prediction head
│   │   └── hierarchical.py       # Person 4: Hierarchical attention aggregator
│   ├── training/
│   │   ├── trainer.py             # Person 5: Training loop
│   │   └── evaluate.py            # Person 5: Metrics and evaluation
│   └── utils/
│       ├── config.py              # Shared config/hyperparameters
│       └── io.py                  # Save/load helpers
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_tfidf.ipynb
│   ├── 03_finbert_experiments.ipynb
│   └── 04_hierarchical_experiments.ipynb
├── experiments/               # W&B logs, saved checkpoints (gitignored)
├── demo/
│   └── app.py                 # HuggingFace Spaces demo (Person 5)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/YOUR_ORG/earnings-sentiment.git
cd earnings-sentiment
pip install -r requirements.txt
```

Open notebooks in Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_ORG/earnings-sentiment/blob/main/notebooks/01_data_exploration.ipynb)

---

## Data Setup

1. Place the local transcript txt dataset under `src/data/NLP_Dataset/`, or set `KAGGLE_TRANSCRIPTS_DIR` in `.env`
2. Copy `.env.example` to `.env` and update paths if needed
3. Run the data pipeline:

```bash
python -m src.data.fetch_transcripts_2   # converts local txt transcripts to JSON
python -m src.data.fetch_returns         # downloads post-call returns
python -m src.data.labels                # constructs labels
```

> ⚠️ Raw data is gitignored. Every team member must run the pipeline locally (or pull from a shared Google Drive link).

---

## Key Design Decisions

**Label definition:** We predict *post-open* abnormal return (starting from morning market open the day after the call), not the overnight gap. This is the more interesting and tradeable signal per project feedback.

**Task formulation:** 3-class classification (up / flat / down) as primary task; regression as secondary. Flat band is ±0.5% abnormal return.

**Transcript chunking:** Each transcript is split into 450-token overlapping chunks (50-token overlap). Chunks encode with FinBERT, then a second-level transformer aggregates chunk representations.

---

## Experiment Tracking

All runs logged to [Weights & Biases](https://wandb.ai). Join the team project:

```python
import wandb
wandb.init(project="earnings-sentiment", entity="YOUR_WANDB_TEAM")
```

---

## Branch Workflow

```
main          ← stable, protected. PR required to merge.
dev           ← integration branch. merge feature branches here first.
feature/data-pipeline
feature/preprocessing
feature/finbert
feature/hierarchical
feature/training-eval
```

---

## Interfaces (READ BEFORE CODING)

The `EarningsDataset` class is the shared contract between the data and model teams.

```python
# src/data/dataset.py — agreed interface
dataset[i] == {
    "chunks":       Tensor[num_chunks, 450],   # token ids per chunk
    "attention_mask": Tensor[num_chunks, 450],
    "label":        int,                        # 0=down, 1=flat, 2=up
    "ticker":       str,
    "call_date":    str,
}
```

**Do not change this interface without a PR and team sign-off.**
