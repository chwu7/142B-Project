"""
src/models/hierarchical.py
Hierarchical transformer — the 'advanced' component

Second-level transformer that aggregates chunk-level FinBERT embeddings
into a call-level representation, then classifies.

Architecture:
    1. ChunkEncoder (from Person 3) → [B, num_chunks, HIDDEN_DIM]
    2. Positional encoding over chunks
    3. Transformer encoder (HIER_NUM_LAYERS layers)
    4. CLS-token or mean-pool → [B, HIDDEN_DIM]
    5. Classification head → [B, NUM_CLASSES]
"""

import torch
import torch.nn as nn
from src.models.finbert_head import ChunkEncoder
from src.utils.config import (
    HIDDEN_DIM, NUM_CLASSES, DROPOUT,
    HIER_NUM_HEADS, HIER_NUM_LAYERS, HIER_FF_DIM,
    MAX_CHUNKS,
)


class ChunkPositionalEncoding(nn.Module):
    """Learnable positional encoding over chunk positions (not token positions)."""
    def __init__(self, max_chunks: int = MAX_CHUNKS):
        super().__init__()
        self.pos_embedding = nn.Embedding(max_chunks + 1, HIDDEN_DIM)  # +1 for CLS

    def forward(self, x):
        """x: [B, seq_len, HIDDEN_DIM], where seq_len may include CLS."""
        B, N, _ = x.shape
        positions = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        return x + self.pos_embedding(positions)


class HierarchicalTransformer(nn.Module):
    """
    Aggregates chunk embeddings using a transformer, then classifies.

    Input:  chunk_embeddings [B, num_chunks, HIDDEN_DIM]  (from ChunkEncoder)
    Output: logits           [B, NUM_CLASSES]
    """
    def __init__(self):
        super().__init__()

        # Learnable [CLS] token prepended to chunk sequence
        self.cls_token = nn.Parameter(torch.randn(1, 1, HIDDEN_DIM))

        self.pos_enc = ChunkPositionalEncoding()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=HIDDEN_DIM,
            nhead=HIER_NUM_HEADS,
            dim_feedforward=HIER_FF_DIM,
            dropout=DROPOUT,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=HIER_NUM_LAYERS
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, HIDDEN_DIM // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(HIDDEN_DIM // 2, NUM_CLASSES),
        )

    def forward(self, chunk_embeddings, chunk_padding_mask=None):
        """
        Args:
            chunk_embeddings:  [B, num_chunks, HIDDEN_DIM]
            chunk_padding_mask: [B, num_chunks] bool — True for padded chunks

        Returns:
            logits: [B, NUM_CLASSES]
        """
        B, N, _ = chunk_embeddings.shape

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)            # [B, 1, HIDDEN_DIM]
        x   = torch.cat([cls, chunk_embeddings], dim=1)   # [B, N+1, HIDDEN_DIM]

        x = self.pos_enc(x)

        # Extend padding mask to account for CLS token (never masked)
        if chunk_padding_mask is not None:
            cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, chunk_padding_mask], dim=1)
        else:
            src_key_padding_mask = None

        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        cls_out = x[:, 0, :]    # CLS token output: [B, HIDDEN_DIM]
        return self.classifier(cls_out)


class HierarchicalModel(nn.Module):
    """
    Full model: ChunkEncoder (Person 3) + HierarchicalTransformer (Person 4).
    Drop-in replacement for BaselineModel.
    """
    def __init__(self, freeze_layers: int = 6):
        super().__init__()
        self.encoder    = ChunkEncoder(freeze_layers=freeze_layers)
        self.aggregator = HierarchicalTransformer()

    def forward(self, chunks, attention_mask):
        """
        Args:
            chunks:         [B, num_chunks, seq_len]
            attention_mask: [B, num_chunks, seq_len]
        Returns:
            logits: [B, NUM_CLASSES]
        """
        B, N, L = chunks.shape

        flat_ids  = chunks.reshape(B * N, L)
        flat_mask = attention_mask.reshape(B * N, L)

        embeddings = self.encoder(flat_ids, flat_mask)  # [B*N, HIDDEN_DIM]
        embeddings = embeddings.reshape(B, N, -1)           # [B,   N, HIDDEN_DIM]

        # Build chunk padding mask: a chunk is padding if all tokens are pad (mask=0)
        chunk_is_pad = (attention_mask.sum(dim=-1) == 0)  # [B, N]

        return self.aggregator(embeddings, chunk_padding_mask=chunk_is_pad)
