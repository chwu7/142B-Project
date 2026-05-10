"""
src/training/trainer.py
OWNER: Person 5 (Training loop, eval, demo)

Training loop with early stopping and W&B logging.

Usage:
    python src/training/trainer.py --model hierarchical --epochs 10
"""
import os
import argparse
import torch
import torch.nn as nn
import wandb
from tqdm import tqdm
from dotenv import load_dotenv

from src.data.dataset import get_dataloaders
from src.training.evaluate import compute_metrics
from src.utils.config import (
    LEARNING_RATE, WEIGHT_DECAY, MAX_EPOCHS, PATIENCE,
    EXPERIMENTS_DIR, NUM_CLASSES,
)
from src.utils.io import save_checkpoint

load_dotenv()


def get_model(model_name: str):
    if model_name == "baseline":
        from src.models.finbert_head import BaselineModel
        return BaselineModel()
    elif model_name == "hierarchical":
        from src.models.hierarchical import HierarchicalModel
        return HierarchicalModel()
    elif model_name == "meanpool":
        from src.models.hierarchical import MeanPoolModel
        return MeanPoolModel()
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(model_name: str = "hierarchical", run_name: str = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Init W&B
    wandb.init(
        project="earnings-sentiment",
        entity=os.getenv("WANDB_ENTITY"),
        name=run_name or model_name,
        config={
            "model": model_name,
            "lr": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "max_epochs": MAX_EPOCHS,
            "patience": PATIENCE,
        },
    )

    model = get_model(model_name).to(device)

    # Differential LRs: FinBERT encoder gets 10x smaller LR to protect pretrained weights
    encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
    encoder_ids    = {id(p) for p in encoder_params}
    head_params    = [p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": LEARNING_RATE * 0.1},
        {"params": head_params,    "lr": LEARNING_RATE},
    ], weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = get_dataloaders()

    best_val_loss = float("inf")
    patience_counter = 0
    ckpt_dir = os.path.join(EXPERIMENTS_DIR, model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    for epoch in range(1, MAX_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} train"):
            chunks = batch["chunks"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            logits = model(chunks, mask)
            loss   = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # ── Validate ──
        val_loss, val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_metrics['accuracy']:.4f}")

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"val_{k}": v for k, v in val_metrics.items()},
        })

        # ── Early stopping ──
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model, optimizer, epoch,
                            {"val_loss": val_loss, **val_metrics},
                            os.path.join(ckpt_dir, "best.pt"))
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break
                
    # ── Final Test Evaluation ──
    test_loss, test_metrics = evaluate(model, test_loader, criterion, device)
    
    print("\nFinal Test Results")
    print(f"test_loss={test_loss:.4f}")
    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}")
    
    wandb.log({
        "test_loss": test_loss,
        **{f"test_{k}": v for k, v in test_metrics.items()},
    })

    wandb.finish()


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in loader:
            chunks = batch["chunks"].to(device)
            mask   = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(chunks, mask)
            loss   = criterion(logits, labels)
            total_loss += loss.item()

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / len(loader)
    metrics  = compute_metrics(all_preds, all_labels)
    return avg_loss, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hierarchical", choices=["baseline", "hierarchical", "meanpool"])
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()
    train(model_name=args.model, run_name=args.run_name)
