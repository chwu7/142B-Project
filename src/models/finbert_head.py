"""
src/models/finbert_head.py
OWNER: Person 3 (FinBERT fine-tuning)

FinBERT chunk encoder + flat prediction head (baseline model).
This encodes each chunk independently — no cross-chunk attention.
Used as the baseline before Person 4 adds hierarchical aggregation.
"""
import torch
import torch.nn as nn
from transformers import AutoModel
from src.utils.config import FINBERT_MODEL, HIDDEN_DIM, NUM_CLASSES, DROPOUT


class ChunkEncoder(nn.Module):
    """
    Wraps FinBERT to encode a single chunk.
    Input:  input_ids [B, seq_len], attention_mask [B, seq_len]
    Output: embedding [B, HIDDEN_DIM]  (CLS token representation)
    """
    def __init__(self, freeze_layers: int = 6):
        """
        Args:
            freeze_layers: number of FinBERT transformer layers to freeze.
                           Freeze early layers to avoid catastrophic forgetting.
                           FinBERT has 12 layers total.
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(FINBERT_MODEL)

        # Freeze the first `freeze_layers` transformer layers
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]  # CLS token: [B, HIDDEN_DIM]


class MeanPoolingHead(nn.Module):
    """
    Baseline aggregation: mean-pool chunk embeddings, then classify.
    Replace this with HierarchicalTransformer from Person 4 for the full model.

    Input:  chunk_embeddings [B, num_chunks, HIDDEN_DIM]
    Output: logits           [B, NUM_CLASSES]
    """
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, NUM_CLASSES),
        )

    def forward(self, chunk_embeddings):
        # Simple mean pooling across chunks
        pooled = chunk_embeddings.mean(dim=1)   # [B, HIDDEN_DIM]
        return self.classifier(pooled)           # [B, NUM_CLASSES]


class BaselineModel(nn.Module):
    """
    Full baseline: ChunkEncoder + MeanPoolingHead.
    One forward pass encodes all chunks in a batch.
    """
    def __init__(self, freeze_layers: int = 6):
        super().__init__()
        self.encoder = ChunkEncoder(freeze_layers=freeze_layers)
        self.head     = MeanPoolingHead()

    def forward(self, chunks, attention_mask):
        """
        Args:
            chunks:         [B, num_chunks, seq_len]
            attention_mask: [B, num_chunks, seq_len]
        Returns:
            logits: [B, NUM_CLASSES]
        """
        B, N, L = chunks.shape

        # Flatten batch × chunks for parallel BERT encoding
        flat_ids  = chunks.view(B * N, L)
        flat_mask = attention_mask.view(B * N, L)

        embeddings = self.encoder(flat_ids, flat_mask)  # [B*N, HIDDEN_DIM]
        embeddings = embeddings.view(B, N, -1)           # [B, N, HIDDEN_DIM]

        return self.head(embeddings)
