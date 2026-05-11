"""
demo/app.py
OWNER: Person 5 (Training loop, eval, demo)

HuggingFace Spaces Gradio demo.
Paste an earnings call transcript → get predicted market reaction.

Deploy:
    1. Create a new Space on huggingface.co (Gradio SDK)
    2. Push this file + requirements.txt + your model checkpoint
    3. Set HF_MODEL_PATH secret in Space settings
"""
import os
import torch
import gradio as gr

# Load model once at startup
MODEL_PATH = os.getenv("HF_MODEL_PATH", "experiments/hierarchical/earnings-sentiment/hierarchical.pt")
LABEL_NAMES = ["📉 Down", "➡️ Flat", "📈 Up"]
LABEL_COLORS = ["#ff4444", "#aaaaaa", "#44bb44"]


def load_model():
    from src.models.hierarchical import HierarchicalModel
    model = HierarchicalModel()
    ckpt  = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


MODEL = None  # lazy load on first predict call


def predict(transcript: str):
    global MODEL
    if MODEL is None:
        MODEL = load_model()

    if not transcript.strip():
        return {name: 0.0 for name in LABEL_NAMES}

    from src.data.chunker import chunk_transcript_to_tensors
    input_ids, attention_mask = chunk_transcript_to_tensors(transcript)

    # Add batch dimension
    input_ids      = input_ids.unsqueeze(0)       # [1, num_chunks, seq_len]
    attention_mask = attention_mask.unsqueeze(0)

    with torch.no_grad():
        logits = MODEL(input_ids, attention_mask)  # [1, 3]
        probs  = torch.softmax(logits, dim=-1)[0]  # [3]

    return {LABEL_NAMES[i]: float(probs[i]) for i in range(3)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        lines=20,
        placeholder="Paste earnings call transcript here...",
        label="Earnings Call Transcript",
    ),
    outputs=gr.Label(
        num_top_classes=3,
        label="Predicted Post-Open Movement",
    ),
    title="Earnings Call Sentiment Analyzer",
    description=(
        "Paste an earnings call transcript to predict how the stock will move "
        "after market open the next day. Powered by FinBERT + Hierarchical Transformer."
    ),
    examples=[
        ["We had a strong quarter with record revenue growth across all segments..."],
        ["We are facing significant headwinds and revising guidance downward..."],
    ],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
