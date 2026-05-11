"""
demo/app.py
OWNER: Person 5 (Training loop, eval, demo)

HuggingFace Spaces Gradio demo.
Paste an earnings call transcript → get predicted market reaction
+ highlighted saliency showing which phrases drove the prediction.

Deploy:
    1. Create a new Space on huggingface.co (Gradio SDK)
    2. Push this file + requirements.txt + your model checkpoint
    3. Set HF_MODEL_PATH secret in Space settings

Explanation method:
    Hierarchical model: two-level attention rollout
      1. Hierarchical transformer CLS-token attention → top-k influential chunks
      2. Gradient × input saliency on FinBERT token embeddings → top phrases per chunk
    Mean pool model: gradient-based chunk scoring
      1. Gradient magnitude through mean pooling → chunk importance scores
      2. Gradient × input saliency on FinBERT token embeddings → top phrases per chunk

    Set MODEL_TYPE = "hierarchical" or "meanpool" below to match your checkpoint.
"""
import os
import re
import html
import torch
import numpy as np
import gradio as gr

MODEL_TYPE  = os.getenv("MODEL_TYPE", "hierarchical")   # "hierarchical" or "meanpool"
MODEL_PATH  = os.getenv("HF_MODEL_PATH", f"experiments/hierarchical/earnings-sentiment/best_meanpool_noearlystop.pt")
LABEL_NAMES = ["📉 Down", "➡️ Flat", "📈 Up"]
LABEL_COLORS = ["#ff4444", "#aaaaaa", "#44bb44"]

# Colour palette for saliency highlights (opacity applied in HTML)
SALIENCY_UP   = "34, 153, 117"   # teal-ish green  → pushed toward Up
SALIENCY_DOWN = "216, 90, 48"    # coral/red        → pushed toward Down
SALIENCY_FLAT = "136, 135, 128"  # neutral gray


# ── Model loading ─────────────────────────────────────────────────────

def load_model():
    # Checkpoint was saved from HierarchicalTransformer directly,
    # so always load HierarchicalModel regardless of MODEL_TYPE
    from src.models.hierarchical import HierarchicalModel
    model = HierarchicalModel()

    ckpt = torch.load(MODEL_PATH, map_location="cpu")
    state_dict = ckpt["model_state_dict"]

    # Remap classifier.* → aggregator.classifier.*
    # and encoder.* stays as encoder.*
    if any(k.startswith("classifier.") for k in state_dict):
        state_dict = {
            ("aggregator." + k if k.startswith("classifier.") else k): v
            for k, v in state_dict.items()
        }

    model.load_state_dict(state_dict)
    model.eval()
    return model



MODEL = None

def get_model():
    global MODEL
    if MODEL is None:
        MODEL = load_model()
    return MODEL


# ── Saliency computation ──────────────────────────────────────────────

def _chunk_attention_scores(model, input_ids, attention_mask):
    """
    Extract CLS-token attention weights from the hierarchical transformer
    to score each chunk's importance.

    Returns: numpy array of shape [num_chunks], summed over heads & layers.
    """
    scores = []

    def _hook(module, input, output):
        # output is (attn_output, attn_weights) when need_weights=True
        # attn_weights: [B, num_heads, seq_len, seq_len]
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            weights = output[1]          # [1, heads, N+1, N+1]
            # CLS (position 0) attention to each chunk position
            cls_attn = weights[0, :, 0, 1:]   # [heads, N]  — exclude CLS→CLS
            scores.append(cls_attn.detach().cpu().mean(dim=0).numpy())

    hooks = []
    for layer in model.aggregator.transformer.layers:
        hooks.append(layer.self_attn.register_forward_hook(_hook))

    with torch.no_grad():
        B, N, L = input_ids.shape
        flat_ids  = input_ids.view(B * N, L)
        flat_mask = attention_mask.view(B * N, L)
        embeddings = model.encoder(flat_ids, flat_mask).view(B, N, -1)
        chunk_is_pad = (attention_mask.sum(dim=-1) == 0)
        model.aggregator(embeddings, chunk_padding_mask=chunk_is_pad)

    for h in hooks:
        h.remove()

    if not scores:
        # Fallback: uniform scores
        return np.ones(N) / N

    # Average across layers, re-normalise
    avg = np.stack(scores, axis=0).mean(axis=0)[:N]
    avg = avg / (avg.sum() + 1e-9)
    return avg


def _chunk_gradient_scores(model, input_ids, attention_mask, predicted_class):
    """
    Score chunks by gradient magnitude through mean pooling.
    Used for BaselineModel (MeanPoolingHead) which has no attention to hook into.

    Returns: numpy array of shape [num_chunks].
    """
    B, N, L = input_ids.shape
    flat_ids  = input_ids.view(B * N, L)
    flat_mask = attention_mask.view(B * N, L)

    with torch.no_grad():
        embeddings_detached = model.encoder(flat_ids, flat_mask).view(B, N, -1)

    embeddings = embeddings_detached.detach().requires_grad_(True)
    pooled = embeddings.mean(dim=1)              # [B, H]
    logits = model.head.classifier(pooled)       # [B, 3]
    logits[0, predicted_class].backward()

    chunk_scores = embeddings.grad[0].norm(dim=-1)   # [N]
    chunk_scores = chunk_scores.detach().cpu().numpy()
    chunk_scores = chunk_scores / (chunk_scores.sum() + 1e-9)
    return chunk_scores


def _get_chunk_scores(model, input_ids, attention_mask, predicted_class):
    """Dispatcher: picks the right chunk scoring method based on MODEL_TYPE."""
    if MODEL_TYPE == "meanpool":
        return _chunk_gradient_scores(model, input_ids, attention_mask, predicted_class)
    else:
        return _chunk_attention_scores(model, input_ids, attention_mask)


def _token_saliency(model, input_ids, attention_mask, target_class):
    """
    Gradient × input saliency at the token embedding level within a single chunk.

    Returns: numpy array of shape [seq_len] with non-negative importance scores.
    """
    embeddings = model.encoder.bert.embeddings(input_ids)   # [1, L, H]
    embeddings = embeddings.detach().requires_grad_(True)

    # Forward from embeddings (skip the embedding lookup layer)
    extended_mask = model.encoder.bert.get_extended_attention_mask(
        attention_mask, input_ids.shape
    )
    hidden = embeddings
    for layer in model.encoder.bert.encoder.layer:
        hidden = layer(hidden, extended_mask)[0]

    cls_vec = hidden[:, 0, :]                        # [1, H]

    if MODEL_TYPE == "meanpool":
        # MeanPoolingHead: single chunk acts as its own pool
        logit = model.head.classifier(cls_vec)[0, target_class]
    else:
        # HierarchicalModel: route CLS through the hierarchical aggregator head
        logit = model.aggregator.classifier(
            model.aggregator.transformer(
                torch.cat([model.aggregator.cls_token, cls_vec.unsqueeze(1)], dim=1)
            )[:, 0, :]
        )[0, target_class]

    logit.backward()

    grad = embeddings.grad[0]                        # [L, H]
    saliency = (grad * embeddings[0].detach()).norm(dim=-1)  # [L]
    return saliency.detach().cpu().numpy()


def compute_saliency(model, input_ids, attention_mask, predicted_class):
    """
    Full two-level saliency pipeline.

    Returns:
        chunk_scores:  np.array [num_chunks]  — importance of each chunk
        token_scores:  list of np.array [seq_len] per chunk
    """
    chunk_scores = _get_chunk_scores(model, input_ids, attention_mask, predicted_class)

    token_scores = []
    num_chunks = input_ids.shape[1]
    for i in range(num_chunks):
        chunk_ids  = input_ids[0, i].unsqueeze(0)    # [1, L]
        chunk_mask = attention_mask[0, i].unsqueeze(0)
        try:
            ts = _token_saliency(model, chunk_ids, chunk_mask, predicted_class)
        except Exception:
            ts = np.zeros(input_ids.shape[-1])
        token_scores.append(ts)

    return chunk_scores, token_scores


# ── Text reconstruction & highlighting ───────────────────────────────

def _tokens_to_words(tokenizer, ids, scores):
    """
    Re-merge wordpiece tokens into words and pool their saliency scores.
    Returns list of (word, score) pairs, excluding special/pad tokens.
    """
    tokens = tokenizer.convert_ids_to_tokens(ids)
    words, word_scores = [], []
    cur_word, cur_scores = "", []

    for tok, sc in zip(tokens, scores):
        if tok in (tokenizer.cls_token, tokenizer.sep_token,
                   tokenizer.pad_token, "[PAD]", "[CLS]", "[SEP]"):
            if cur_word:
                words.append(cur_word)
                word_scores.append(float(np.mean(cur_scores)))
                cur_word, cur_scores = "", []
            continue

        if tok.startswith("##"):
            cur_word += tok[2:]
            cur_scores.append(sc)
        else:
            if cur_word:
                words.append(cur_word)
                word_scores.append(float(np.mean(cur_scores)))
            cur_word = tok
            cur_scores = [sc]

    if cur_word:
        words.append(cur_word)
        word_scores.append(float(np.mean(cur_scores)))

    return list(zip(words, word_scores))


def build_highlighted_html(
    model, tokenizer, input_ids, attention_mask,
    chunk_scores, token_scores, predicted_class, top_chunks=5
):
    """
    Reconstruct transcript HTML with saliency-coloured highlights.

    Colour logic:
        - Chunks ranked by chunk_scores; top_chunks get token-level highlights.
        - Within top chunks, token saliency controls opacity (0.1 – 0.9).
        - Colour encodes predicted class: Up → green, Down → red, Flat → gray.
    """
    from src.data.chunker import get_tokenizer  # reuse cached tokenizer

    num_chunks = input_ids.shape[1]
    top_indices = set(np.argsort(chunk_scores)[-top_chunks:])

    # Normalise token scores globally so colours are comparable across chunks
    all_scores = np.concatenate(token_scores)
    sc_max = all_scores.max() + 1e-9

    color_rgb = {0: SALIENCY_DOWN, 1: SALIENCY_FLAT, 2: SALIENCY_UP}[predicted_class]

    parts = []
    for i in range(num_chunks):
        ids_i  = input_ids[0, i].tolist()
        ts_i   = token_scores[i]
        word_scores = _tokens_to_words(tokenizer, ids_i, ts_i)

        if i not in top_indices:
            # Render without highlights (plain text, slightly dimmed)
            text = " ".join(w for w, _ in word_scores)
            parts.append(f'<span class="chunk-dim">{html.escape(text)} </span>')
        else:
            chunk_importance = chunk_scores[i]
            span_parts = []
            for word, sc in word_scores:
                opacity = 0.08 + 0.82 * min(sc / sc_max, 1.0)
                span_parts.append(
                    f'<mark style="background:rgba({color_rgb},{opacity:.2f});'
                    f'border-radius:2px;padding:0 1px;" '
                    f'title="saliency: {sc:.3f}">'
                    f'{html.escape(word)}</mark>'
                )
            parts.append(
                f'<span class="chunk-top" data-score="{chunk_importance:.3f}">'
                + " ".join(span_parts)
                + " </span>"
            )

    return "".join(parts)


def extract_top_phrases(model, tokenizer, input_ids, attention_mask,
                        chunk_scores, token_scores, n=8, window=6):
    """
    Extract the top-n most salient contiguous word windows across all chunks.

    Returns list of (phrase_text, score, chunk_rank) sorted by score descending.
    """
    num_chunks = input_ids.shape[1]
    all_scores_flat = np.concatenate(token_scores)
    sc_max = all_scores_flat.max() + 1e-9

    phrases = []
    chunk_ranks = np.argsort(chunk_scores)[::-1]  # highest chunk first

    for rank, i in enumerate(chunk_ranks[:8]):   # only look in top 8 chunks
        ids_i = input_ids[0, i].tolist()
        ts_i  = token_scores[i]
        word_scores = _tokens_to_words(tokenizer, ids_i, ts_i)

        if not word_scores:
            continue

        words  = [w for w, _ in word_scores]
        scores = np.array([s for _, s in word_scores])

        # Sliding window over words
        for start in range(len(words) - window + 1):
            end   = start + window
            score = scores[start:end].mean() / sc_max
            phrase = " ".join(words[start:end])
            # Skip very short or punctuation-heavy phrases
            alpha_ratio = sum(c.isalpha() for c in phrase) / (len(phrase) + 1)
            if alpha_ratio > 0.55:
                phrases.append((phrase, float(score), rank + 1))

    # Deduplicate (greedy overlap removal) and take top-n
    phrases.sort(key=lambda x: -x[1])
    selected, seen_words = [], set()
    for phrase, score, chunk_rank in phrases:
        phrase_words = set(phrase.lower().split())
        if len(phrase_words & seen_words) / len(phrase_words) < 0.5:
            selected.append((phrase, score, chunk_rank))
            seen_words.update(phrase_words)
        if len(selected) >= n:
            break

    return selected


# ── Gradio interface ──────────────────────────────────────────────────

def predict_and_explain(transcript: str):
    """
    Main Gradio handler. Returns:
        - probabilities dict for Label widget
        - HTML string with highlighted transcript
        - HTML string with top-phrases panel
    """
    if not transcript.strip():
        empty = {name: 0.0 for name in LABEL_NAMES}
        return empty, "<p style='color:gray'>Paste a transcript above.</p>", ""

    model = get_model()

    from src.data.chunker import chunk_transcript_to_tensors, get_tokenizer
    tokenizer = get_tokenizer()
    input_ids, attention_mask = chunk_transcript_to_tensors(transcript)
    input_ids      = input_ids.unsqueeze(0)       # [1, num_chunks, seq_len]
    attention_mask = attention_mask.unsqueeze(0)

    # ── Prediction ────────────────────────────────────────────────────
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
        probs  = torch.softmax(logits, dim=-1)[0]

    predicted_class = int(probs.argmax())
    prob_dict = {LABEL_NAMES[i]: float(probs[i]) for i in range(3)}

    # ── Saliency ──────────────────────────────────────────────────────
    try:
        chunk_scores, token_scores = compute_saliency(
            model, input_ids, attention_mask, predicted_class
        )
    except Exception as e:
        print(f"Saliency computation failed: {e}")
        highlighted_html = "<p style='color:gray'>Saliency unavailable for this transcript.</p>"
        phrases_html = ""
        return prob_dict, highlighted_html, phrases_html

    # ── Build highlighted transcript ──────────────────────────────────
    top_chunks = 7 if MODEL_TYPE == "meanpool" else 5
    highlighted = build_highlighted_html(
        model, tokenizer, input_ids, attention_mask,
        chunk_scores, token_scores, predicted_class, top_chunks=top_chunks
    )

    color_hex = LABEL_COLORS[predicted_class]
    label_name = LABEL_NAMES[predicted_class]

    highlighted_html = f"""
<style>
  .transcript-container {{
    font-family: Georgia, serif;
    font-size: 14px;
    line-height: 1.85;
    color: #1a1a1a;
    max-height: 480px;
    overflow-y: auto;
    padding: 1rem 1.25rem;
    border: 0.5px solid #ddd;
    border-radius: 8px;
    background: #fafafa;
  }}
  .chunk-dim {{ color: #888; }}
  .chunk-top {{ color: #1a1a1a; }}
  .legend {{
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; color: #555;
    margin-bottom: 10px; font-family: sans-serif;
  }}
  .legend-swatch {{
    width: 18px; height: 10px;
    border-radius: 2px;
    background: rgba({color_rgb},{0.65});
  }}
</style>
<div class="legend">
  <span>Highlights show phrases driving prediction toward <strong>{label_name}</strong>.</span>
  <div class="legend-swatch"></div>
  <span>Brighter = more influential. Dimmed text = lower-impact sections.</span>
</div>
<div class="transcript-container">{highlighted}</div>
""".replace("{color_rgb}", {0: SALIENCY_DOWN, 1: SALIENCY_FLAT, 2: SALIENCY_UP}[predicted_class])

    # ── Build top-phrases panel ───────────────────────────────────────
    top_phrases = extract_top_phrases(
        model, tokenizer, input_ids, attention_mask,
        chunk_scores, token_scores, n=8, window=6
    )

    if not top_phrases:
        phrases_html = "<p style='color:gray;font-family:sans-serif;font-size:13px;'>No key phrases extracted.</p>"
    else:
        color_rgb_val = {0: SALIENCY_DOWN, 1: SALIENCY_FLAT, 2: SALIENCY_UP}[predicted_class]
        rows = ""
        for rank, (phrase, score, chunk_rank) in enumerate(top_phrases, 1):
            bar_width = int(score * 100)
            rows += f"""
<div style="margin-bottom:12px;">
  <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px;">
    <span style="font-size:13px;font-family:Georgia,serif;color:#1a1a1a;">
      {rank}. &ldquo;{html.escape(phrase)}&rdquo;
    </span>
    <span style="font-size:11px;color:#888;font-family:sans-serif;margin-left:8px;white-space:nowrap;">
      score {score:.2f}
    </span>
  </div>
  <div style="background:#eee;border-radius:3px;height:5px;overflow:hidden;">
    <div style="width:{bar_width}%;height:5px;
                background:rgba({color_rgb_val},0.75);
                border-radius:3px;transition:width 0.3s;"></div>
  </div>
</div>"""

        phrases_html = f"""
<style>
  .phrases-box {{
    padding: 1rem 1.25rem;
    border: 0.5px solid #ddd;
    border-radius: 8px;
    background: #fafafa;
  }}
  .phrases-heading {{
    font-family: sans-serif;
    font-size: 12px;
    font-weight: 500;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-bottom: 14px;
  }}
</style>
<div class="phrases-box">
  <div class="phrases-heading">Top influential phrases</div>
  {rows}
</div>"""

    return prob_dict, highlighted_html, phrases_html


# ── Build Gradio UI ───────────────────────────────────────────────────

model_desc = "FinBERT + Mean Pool (gradient saliency)" if MODEL_TYPE == "meanpool" \
    else "FinBERT + Hierarchical Transformer (attention saliency)"

with gr.Blocks(title="Earnings Call Sentiment Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        f"""
        # Earnings Call Sentiment Analyzer
        Paste an earnings call transcript to predict post-open stock movement.
        The **Explanation** tab shows which phrases most influenced the prediction.

        *Powered by {model_desc}.*
        """
    )

    with gr.Row():
        transcript_input = gr.Textbox(
            lines=18,
            placeholder="Paste earnings call transcript here...",
            label="Earnings Call Transcript",
        )

    with gr.Row():
        run_btn = gr.Button("Analyze Transcript", variant="primary")

    with gr.Row():
        prediction_output = gr.Label(
            num_top_classes=3,
            label="Predicted Post-Open Movement",
        )

    with gr.Tabs():
        with gr.Tab("Highlighted Transcript"):
            gr.Markdown(
                "Sections that most influenced the prediction are **colour-highlighted**. "
                "Brighter = stronger signal. Dimmed text had less influence."
            )
            highlighted_output = gr.HTML(label="Saliency-highlighted transcript")

        with gr.Tab("Top Phrases"):
            gr.Markdown(
                "The most influential contiguous phrases extracted from the transcript, "
                "ranked by saliency score and shown with a relative importance bar."
            )
            phrases_output = gr.HTML(label="Key phrases")

    gr.Examples(
        examples=[
            ["We had a record quarter with revenue up 18% year-over-year. "
             "Demand across cloud and AI workloads exceeded expectations and we are "
             "raising full-year guidance. Operating margins expanded significantly "
             "and free cash flow hit an all-time high."],
            ["We are revising guidance downward due to weakening macro conditions. "
             "Supply chain disruptions have pressured gross margins materially and "
             "we expect continued headwinds in the back half. Customer spending "
             "is under increasing scrutiny and deal cycles have lengthened."],
        ],
        inputs=transcript_input,
        label="Example transcripts",
    )

    run_btn.click(
        fn=predict_and_explain,
        inputs=transcript_input,
        outputs=[prediction_output, highlighted_output, phrases_output],
    )

if __name__ == "__main__":
    demo.launch()




