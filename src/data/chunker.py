"""
src/data/chunker.py
OWNER: Person 2 (Preprocessing)

Splits a raw transcript string into overlapping token chunks
suitable for FinBERT (max 512 tokens, we use 450 + 50 overlap).
"""
import transformers
from transformers import AutoTokenizer
from src.utils.config import FINBERT_MODEL, CHUNK_SIZE, CHUNK_OVERLAP, MAX_CHUNKS

# Suppress the "sequence length > 512" warning — we intentionally tokenize the
# full transcript before slicing it into chunks ourselves.
transformers.logging.set_verbosity_error()


_tokenizer = None

def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
    return _tokenizer


def chunk_transcript(text: str) -> list[dict]:
    """
    Splits transcript text into overlapping token chunks.

    Returns list of dicts:
        [{"input_ids": [...], "attention_mask": [...]}, ...]

    Each chunk has exactly CHUNK_SIZE tokens (padded if needed).
    """
    if not text or not text.strip():
        raise ValueError("chunk_transcript received empty or whitespace-only text")

    tokenizer = get_tokenizer()

    # Tokenize the full transcript without truncation
    encoding = tokenizer(
        text,
        add_special_tokens=False,
        return_tensors=None,
    )
    input_ids      = encoding["input_ids"]
    attention_mask = encoding["attention_mask"]

    chunks = []
    step   = CHUNK_SIZE - CHUNK_OVERLAP
    start  = 0

    while start < len(input_ids) and len(chunks) < MAX_CHUNKS:
        end = start + CHUNK_SIZE

        chunk_ids  = input_ids[start:end]
        chunk_mask = attention_mask[start:end]

        # Pad to CHUNK_SIZE if this is a short final chunk
        pad_len = CHUNK_SIZE - len(chunk_ids)
        chunk_ids  += [tokenizer.pad_token_id] * pad_len
        chunk_mask += [0] * pad_len

        chunks.append({
            "input_ids":      chunk_ids,
            "attention_mask": chunk_mask,
        })
        start += step

    return chunks


def chunk_transcript_to_tensors(text: str):
    """
    Convenience wrapper — returns (input_ids, attention_mask) as tensors.
    Shape: [num_chunks, CHUNK_SIZE]
    """
    import torch
    chunks = chunk_transcript(text)
    input_ids      = torch.tensor([c["input_ids"]      for c in chunks])
    attention_mask = torch.tensor([c["attention_mask"] for c in chunks])
    return input_ids, attention_mask
