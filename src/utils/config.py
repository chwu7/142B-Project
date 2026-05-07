"""
src/utils/config.py
Shared hyperparameters and constants. Change here, affects everywhere.
"""

# ── Data ──────────────────────────────────────────────────────────────
TICKERS = None          # None = all S&P 500; or list like ["AAPL", "MSFT"]
DATE_START = "2015-01-01"
DATE_END   = "2024-12-31"

# Label construction
FLAT_BAND = 0.005       # ±0.5% abnormal return = "flat" class
RETURN_WINDOW_DAYS = 2  # days after market open to measure return

# ── Chunking ──────────────────────────────────────────────────────────
CHUNK_SIZE    = 450     # tokens per chunk (leaves room for [CLS]/[SEP])
CHUNK_OVERLAP = 50      # overlapping tokens between adjacent chunks
MAX_CHUNKS    = 32      # cap per transcript to limit memory

# ── Model ─────────────────────────────────────────────────────────────
FINBERT_MODEL = "ProsusAI/finbert"
HIDDEN_DIM    = 768     # FinBERT hidden size
NUM_CLASSES   = 3       # 0=down, 1=flat, 2=up
DROPOUT       = 0.1

# Hierarchical transformer
HIER_NUM_HEADS  = 8
HIER_NUM_LAYERS = 2
HIER_FF_DIM     = 1024

# ── Training ──────────────────────────────────────────────────────────
BATCH_SIZE      = 16
NUM_WORKERS     = 2
LEARNING_RATE   = 2e-5
WEIGHT_DECAY    = 1e-2
MAX_EPOCHS      = 10
PATIENCE        = 3     # early stopping patience
TRAIN_SPLIT     = 0.7
VAL_SPLIT       = 0.15
TEST_SPLIT      = 0.15

# ── Paths ─────────────────────────────────────────────────────────────
import os
ROOT_DIR        = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_RAW_DIR    = os.path.join(ROOT_DIR, "src", "data", "raw")
DATA_PROC_DIR   = os.path.join(ROOT_DIR, "src", "data", "processed")
EXPERIMENTS_DIR = os.path.join(ROOT_DIR, "experiments")
