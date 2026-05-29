"""
Inference for the CLIP-style CIFAR-100 visual encoder.

Model checkpoint is hosted on HuggingFace:
    https://huggingface.co/haripra1112001/clip-cifar100-mobilenet

Usage:
    # Simple
    python inference.py --image my_image.jpg

    # With TTA (better accuracy)
    python inference.py --image my_image.jpg --tta
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from huggingface_hub import hf_hub_download

from image_helper import ImageEncoder


# =============================================================================
# LOAD
# =============================================================================

def load_model(device="cpu"):
    """Download checkpoint from HuggingFace and load model + text embeddings."""
    path = hf_hub_download(
        repo_id="haripra1112001/clip-cifar100-mobilenet",
        filename="best_cifar100_projection.pth",
    )
    ckpt = torch.load(path, map_location=device, weights_only=False)

    model = ImageEncoder(proj_dim=128, device=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    text_emb    = ckpt["text_embeddings"].numpy()   # (100, 128)
    class_words = ckpt["class_words"]               # list of 100 names
    text_emb_norm = text_emb / np.linalg.norm(text_emb, axis=1, keepdims=True)

    return model, text_emb_norm, class_words


# =============================================================================
# TRANSFORMS
# =============================================================================

def _tta_transforms():
    """8 deterministic TTA transforms (center crop, flips, multiple scales)."""
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return [
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224),
                             transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(240), transforms.CenterCrop(224),
                             transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(248), transforms.CenterCrop(224),
                             transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(232), transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(240), transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), norm]),
        transforms.Compose([transforms.Resize(248), transforms.CenterCrop(224),
                             transforms.RandomHorizontalFlip(p=1.0),
                             transforms.ToTensor(), norm]),
    ]


# =============================================================================
# INFERENCE
# =============================================================================

def predict(image_path, model, text_emb_norm, class_words, top_k=5):
    """
    Simple single-view inference.

    Args:
        image_path: path to image file
        model: loaded ImageEncoder (eval mode)
        text_emb_norm: np.ndarray (100, 128), L2-normalised
        class_words: list of 100 class name strings
        top_k: number of top predictions to return

    Returns:
        list of (class_name, score) tuples
    """
    preprocess = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        _, proj = model(img)
    img_emb = F.normalize(proj, p=2, dim=1).numpy()    # (1, 128)
    sims    = (text_emb_norm @ img_emb.T).flatten()    # (100,)
    top_idx = np.argsort(-sims)[:top_k]
    return [(class_words[i], round(float(sims[i]), 3)) for i in top_idx]


def predict_tta(image_path, model, text_emb_norm, class_words, top_k=5, device="cpu"):
    """
    8-view Test-Time Augmentation inference (higher accuracy, ~8× slower).

    Averages raw projection vectors across all 8 views before normalising —
    this is geometrically cleaner than averaging normalised vectors.

    Args:
        image_path: path to image file
        model: loaded ImageEncoder (eval mode)
        text_emb_norm: np.ndarray (100, 128), L2-normalised
        class_words: list of 100 class name strings
        top_k: number of top predictions to return
        device: 'cpu' or 'cuda'

    Returns:
        list of (class_name, score) tuples
    """
    pil = Image.open(image_path).convert("RGB")
    tta_batch = torch.stack([t(pil) for t in _tta_transforms()]).to(device)

    with torch.no_grad():
        _, visual_proj = model(tta_batch)                           # (8, 128)

    # Average UNNORMALISED, then normalise once
    avg_emb = F.normalize(visual_proj.mean(dim=0, keepdim=True), p=2, dim=1)
    img_emb = avg_emb.cpu().numpy()                                 # (1, 128)

    sims    = (text_emb_norm @ img_emb.T).flatten()                 # (100,)
    top_idx = np.argsort(-sims)[:top_k]
    return [(class_words[i], round(float(sims[i]), 3)) for i in top_idx]


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CIFAR-100 visual encoder inference")
    parser.add_argument("--image",  required=True, help="Path to input image")
    parser.add_argument("--tta",    action="store_true", help="Use 8-view TTA (slower, more accurate)")
    parser.add_argument("--top_k",  type=int, default=5, help="Number of top predictions (default: 5)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    args = parser.parse_args()

    print("Loading model from HuggingFace...")
    model, text_emb_norm, class_words = load_model(device=args.device)

    if args.tta:
        print(f"Running TTA inference on: {args.image}")
        results = predict_tta(args.image, model, text_emb_norm, class_words,
                              top_k=args.top_k, device=args.device)
    else:
        print(f"Running inference on: {args.image}")
        results = predict(args.image, model, text_emb_norm, class_words,
                          top_k=args.top_k)

    print(f"\nTop {args.top_k} predictions:")
    for rank, (name, score) in enumerate(results, 1):
        print(f"  {rank}. {name:20s}  {score:.3f}")
