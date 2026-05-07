"""
src/data/dataset.py
OWNER: Person 2 (Preprocessing)

PyTorch Dataset class for earnings call transcripts.
THIS IS THE SHARED INTERFACE — do not change the __getitem__ contract
without a PR and team sign-off.

Each item:
    {
        "chunks":         Tensor[num_chunks, CHUNK_SIZE],  # input_ids
        "attention_mask": Tensor[num_chunks, CHUNK_SIZE],
        "label":          int,                              # 0=down, 1=flat, 2=up
        "ticker":         str,
        "call_date":      str,
    }
"""
import os
import json
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, Subset
from src.data.chunker import chunk_transcript_to_tensors
from src.utils.config import (
    DATA_RAW_DIR, DATA_PROC_DIR,
    TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    BATCH_SIZE, NUM_WORKERS,
)

TRANSCRIPTS_DIR = os.path.join(DATA_RAW_DIR, "transcripts")
INDEX_PATH      = os.path.join(DATA_PROC_DIR, "master_index.parquet")


class EarningsDataset(Dataset):
    def __init__(self, index_path: str = INDEX_PATH, cache: bool = True):
        """
        Args:
            index_path: path to master_index.parquet (from labels.py)
            cache: if True, caches chunked tensors in data/processed/cache/
        """
        self.index = pd.read_parquet(index_path)
        self.cache = cache
        self.cache_dir = os.path.join(DATA_PROC_DIR, "cache")
        if cache:
            os.makedirs(self.cache_dir, exist_ok=True)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row = self.index.iloc[idx]
        ticker    = row["ticker"]
        call_date = str(row["call_date"])[:10]  # coerce Timestamp → "YYYY-MM-DD"
        label     = int(row["label"])

        # Try cache first
        cache_key  = f"{ticker}_{call_date}"
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.pt")

        if self.cache and os.path.exists(cache_path):
            cached = torch.load(cache_path, weights_only=False)
            chunks = cached["chunks"]
            mask   = cached["attention_mask"]
        else:
            transcript_path = os.path.join(
                TRANSCRIPTS_DIR, f"{ticker}_{call_date}.json"
            )
            with open(transcript_path) as f:
                data = json.load(f)
            text = data["content"]

            chunks, mask = chunk_transcript_to_tensors(text)

            if self.cache:
                torch.save({"chunks": chunks, "attention_mask": mask}, cache_path)

        return {
            "chunks":         chunks,        # [num_chunks, CHUNK_SIZE]
            "attention_mask": mask,          # [num_chunks, CHUNK_SIZE]
            "label":          label,         # int
            "ticker":         ticker,        # str
            "call_date":      call_date,     # str
        }


def collate_fn(batch):
    """
    Custom collate: pads num_chunks to the max in the batch.
    Returns tensors of shape [B, max_chunks, CHUNK_SIZE].
    """
    import torch.nn.functional as F
    max_chunks = max(item["chunks"].shape[0] for item in batch)

    chunks_padded = []
    mask_padded   = []
    labels        = []
    tickers       = []
    dates         = []

    for item in batch:
        n = item["chunks"].shape[0]
        pad = max_chunks - n
        chunks_padded.append(F.pad(item["chunks"], (0, 0, 0, pad)))
        mask_padded.append(F.pad(item["attention_mask"], (0, 0, 0, pad)))
        labels.append(item["label"])
        tickers.append(item["ticker"])
        dates.append(item["call_date"])

    return {
        "chunks":         torch.stack(chunks_padded),   # [B, max_chunks, CHUNK_SIZE]
        "attention_mask": torch.stack(mask_padded),
        "label":          torch.tensor(labels),
        "ticker":         tickers,
        "call_date":      dates,
    }


def get_dataloaders(index_path: str = INDEX_PATH, seed: int = 42):
    """
    Returns (train_loader, val_loader, test_loader) with chronological splits.
    Sorted by call_date so train = oldest, test = most recent (no leakage).
    """
    dataset = EarningsDataset(index_path=index_path)
    # Sort by date so splits are chronological, not random
    sorted_idx = dataset.index["call_date"].argsort().tolist()
    n       = len(sorted_idx)
    n_train = int(n * TRAIN_SPLIT)
    n_val   = int(n * VAL_SPLIT)

    train_indices = sorted_idx[:n_train]
    val_indices   = sorted_idx[n_train:n_train + n_val]
    test_indices  = sorted_idx[n_train + n_val:]

    train_set = Subset(dataset, train_indices)
    val_set   = Subset(dataset, val_indices)
    test_set  = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=NUM_WORKERS)

    return train_loader, val_loader, test_loader
