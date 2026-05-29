"""
Training script for the CLIP-style CIFAR-100 visual encoder.

Trains a MobileNetV3-Small + 4-layer projection head with contrastive
learning against Skip-Gram text embeddings of CIFAR-100 class names.

You need pre-trained Skip-Gram text embeddings to run this.
A trained set (523-word vocabulary, 128-dim) is available on HuggingFace:
    https://huggingface.co/haripra1112001/visual-skipgram-cifar100

Pre-trained image encoder checkpoint:
    https://huggingface.co/haripra1112001/clip-cifar100-mobilenet

Usage:
    python train.py --text_emb text_embeddings.npy
    python train.py --text_emb text_embeddings.npy --device cuda --save_path my_model.pth
"""

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from torchvision import datasets

from image_helper import (
    ImageEncoder,
    CIFAR100Filtered,
    filter_dataset_indices,
    create_data_splits,
    create_dataloaders,
    train_with_early_stopping,
)

# =============================================================================
# CONFIG  (matches the published checkpoint)
# =============================================================================

CONFIG = {
    "proj_dim":           128,
    "lr":                 2.5e-4,
    "weight_decay":       0.007,
    "temperature":        0.07,
    "epochs":             200,
    "patience":           40,
    "batch_sizes":        {"train": 256, "eval": 512},
    "accumulation_steps": 2,
    "save_path":          "best_cifar100_projection.pth",
    "data_root":          "./data",
    "seed":               42,
}

# =============================================================================
# HELPERS
# =============================================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--text_emb", required=True,
        help="Path to .npy or .pt file with text embeddings (vocab_size, 128). "
             "Download from https://huggingface.co/haripra1112001/visual-skipgram-cifar100",
    )
    parser.add_argument(
        "--class_words", default=None,
        help="Plain-text file with class names (one per line), matching rows of text_emb. "
             "Defaults to CIFAR-100 fine-label names if omitted.",
    )
    parser.add_argument("--device",    default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", default=CONFIG["save_path"])
    parser.add_argument("--epochs",    type=int, default=CONFIG["epochs"])
    parser.add_argument("--patience",  type=int, default=CONFIG["patience"])
    parser.add_argument("--data_root", default=CONFIG["data_root"])
    args = parser.parse_args()

    set_seed(CONFIG["seed"])
    device = torch.device(args.device)

    # ---- text embeddings ----------------------------------------
    if args.text_emb.endswith(".npy"):
        text_emb = torch.from_numpy(np.load(args.text_emb)).float()
    else:
        text_emb = torch.load(args.text_emb, map_location="cpu").float()

    # ---- class names --------------------------------------------
    base_ds = datasets.CIFAR100(root=args.data_root, train=True, download=True)
    if args.class_words:
        with open(args.class_words) as f:
            class_words = [l.strip() for l in f if l.strip()]
    else:
        class_words = base_ds.classes       # 100 fine-label names

    assert len(class_words) == text_emb.shape[0], (
        f"class_words ({len(class_words)}) != text_emb rows ({text_emb.shape[0]})"
    )
    label_to_word    = {i: class_words[i] for i in range(len(class_words))}
    label_to_emb_idx = {i: i for i in range(len(class_words))}

    # ---- data splits --------------------------------------------
    train_full = CIFAR100Filtered(root=args.data_root, split="train")
    test_full  = CIFAR100Filtered(root=args.data_root, split="val")

    all_idx  = filter_dataset_indices(train_full, label_to_emb_idx)
    test_idx = filter_dataset_indices(test_full,  label_to_emb_idx)
    train_idx, val_idx = create_data_splits(all_idx, val_ratio=0.2, seed=42)

    print(f"Train: {len(train_idx):,}  Val: {len(val_idx):,}  Test: {len(test_idx):,}")

    config = {**CONFIG, "save_path": args.save_path, "epochs": args.epochs, "patience": args.patience}
    dataloaders = create_dataloaders(train_idx, val_idx, test_idx, config["batch_sizes"])

    # ---- model --------------------------------------------------
    model     = ImageEncoder(proj_dim=config["proj_dim"], device=str(device)).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {total:,} total params, {trainable:,} trainable")

    # ---- train --------------------------------------------------
    text_emb_dev = F.normalize(text_emb, p=2, dim=1).to(device)
    history, best_epoch, best_sim, best_loss = train_with_early_stopping(
        model, dataloaders, text_emb_dev, class_words, label_to_word, config, device
    )

    print(f"\nTraining complete.")
    print(f"  Best epoch:           {best_epoch}")
    print(f"  Best val similarity:  {best_sim:.4f}")
    print(f"  Checkpoint saved to:  {config['save_path']}")


if __name__ == "__main__":
    main()
