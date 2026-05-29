import random
import requests
from io import BytesIO
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.transforms.v2 import Resize, ToTensor, Normalize, Compose, ToDtype
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

import unittest
import tempfile
import os
from unittest.mock import Mock, MagicMock, patch
from torch.utils.data import TensorDataset


# =============================================================================
# DATASET & MODEL
# =============================================================================

class CIFAR100Filtered(Dataset):
    """
    CIFAR-100 dataset wrapper with preprocessing and train/val split support.
    
    Args:
        root (str): Directory to store/load CIFAR-100 data
        split (str): Either "train" or "val" to specify which split to use
        transform (callable, optional): Transform to apply to images. If None, uses default.
    
    Attributes:
        dataset: The underlying torchvision CIFAR100 dataset
    """
    
    def __init__(self, root="./data", split="train", transform=None):
        assert split in ("train", "val"), "split must be in 'train' or 'val'"

        if transform is None:
            if split == "train":
                transform = Compose([
                    transforms.Resize(256),
                    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandAugment(num_ops=2, magnitude=6),
                    ToTensor(),
                    transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
                    Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ])
            else:
                # Validation/test transform (no changes)
                transform = Compose([
                    Resize(224),
                    ToTensor(),
                    Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    )
                ])

        self.dataset = datasets.CIFAR100(root=root, train=(split=="train"), download=True, transform=transform)
            
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class ImageEncoder(nn.Module):
    """
    MobileNetV3-based image encoder with PARTIAL BACKBONE UNFREEZING.
    """
    
    def __init__(self, proj_dim=128, device="cuda"):
        super().__init__()
        self.device = device
        
        # Load pretrained MobileNetV3-Small
        base = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-1]).to(device)
        
        #Freeze all layers first
        for p in self.backbone.parameters():
            p.requires_grad = False
        
        total_layers = len(list(self.backbone[0].children()))

        # Unfreeze last 3 inverted residual blocks
        for i, layer in enumerate(self.backbone[0].children()):
            if i >= total_layers - 3:
                for p in layer.parameters():
                    p.requires_grad = True

        # Count trainable parameters
        backbone_trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        backbone_total = sum(p.numel() for p in self.backbone.parameters())
 
        self.projection = nn.Sequential(
            # Layer 1: 576 → 1024
            nn.Linear(576, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            # Layer 2: 1024 → 512
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.15),
            
            # Layer 3: 512 → 256
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            
            # Layer 4: 256 → 128
            nn.Linear(256, proj_dim)
        ).to(device)
        
        projection_params = sum(p.numel() for p in self.projection.parameters())

    def forward(self, x):
        """
        Forward pass with PARTIALLY TRAINABLE backbone.
        
        Args:
            x: Input images [batch_size, 3, 224, 224]
            
        Returns:
            feats: Backbone features [batch_size, 576]
            out: Projected embeddings [batch_size, proj_dim]
        """
        # No torch.no_grad() - as last 2 layers need gradients
        feats = self.backbone(x).flatten(1)  # [batch, 576]
        out = self.projection(feats)          # [batch, 128]
        return feats, out

# =============================================================================
# DATA & TRAINING UTILITIES
# =============================================================================

def filter_dataset_indices(dataset, valid_labels):
    """Return indices of samples with labels in valid_labels set."""
    return [i for i, label in enumerate(dataset.dataset.targets) if label in valid_labels]

def create_data_splits(indices, val_ratio=0.2, seed=42):
    """Split indices into train/val sets."""
    np.random.seed(seed)
    indices = np.array(indices)
    np.random.shuffle(indices)
    split_idx = int((1 - val_ratio) * len(indices))
    return indices[:split_idx].tolist(), indices[split_idx:].tolist()

#Module-level worker init function
def _worker_init_fn(worker_id):
    """Worker initialization for reproducibility."""
    import random
    import numpy as np
    np.random.seed(42 + worker_id)
    random.seed(42 + worker_id)

def create_dataloaders(train_idx, val_idx, test_idx, batch_sizes):
    """Create train, val, and test dataloaders with multi-worker loading."""
    datasets_dict = {
        'train': Subset(CIFAR100Filtered(split="train"), train_idx),
        'val': Subset(CIFAR100Filtered(split="train"), val_idx), 
        'test': Subset(CIFAR100Filtered(split="val"), test_idx)
    }
    
    # Multi-worker loading + pin_memory + persistent workers
    return {
        k: DataLoader(
            v, 
            batch_size=batch_sizes['train' if k == 'train' else 'eval'], 
            shuffle=(k == 'train'),
            num_workers=4,
            pin_memory=True,
            persistent_workers=True,
            worker_init_fn=_worker_init_fn,
            generator=torch.Generator().manual_seed(42)
        ) 
        for k, v in datasets_dict.items()
    }

def compute_contrastive_loss(visual_proj, text_emb, temperature, smoothing=0.1):
    """
    Compute symmetric InfoNCE loss with optional label smoothing.
    
    Args:
        visual_proj: Visual embeddings [B, D]
        text_emb: Text embeddings [B, D]
        temperature: Temperature parameter for scaling
        smoothing: Label smoothing factor (0.1 = 10% smoothing)
    
    Returns:
        Symmetric contrastive loss
    """
    V = F.normalize(visual_proj, p=2, dim=1)
    S = F.normalize(text_emb, p=2, dim=1)
    logits = torch.matmul(V, S.T) / temperature
    
    batch_size = logits.shape[0]
    labels = torch.arange(batch_size, device=visual_proj.device)
    
    if smoothing > 0:
        # Create soft labels
        soft_labels = torch.zeros_like(logits)
        soft_labels.fill_(smoothing / (batch_size - 1))  # Distribute smoothing to negatives
        soft_labels.scatter_(1, labels.unsqueeze(1), 1.0 - smoothing)  # Positive gets (1 - smoothing)
        
        # Cross entropy with soft labels
        log_probs = F.log_softmax(logits, dim=1)
        loss_i2t = -(soft_labels * log_probs).sum(dim=1).mean()
        
        log_probs_t = F.log_softmax(logits.T, dim=1)
        loss_t2i = -(soft_labels.T * log_probs_t).sum(dim=1).mean()
        
        return (loss_i2t + loss_t2i) / 2
    else:
        # Standard cross entropy (backward compatible)
        return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def run_epoch(model, dataloader, text_emb, class_words, label_to_word, optimizer, temperature, device, mode='train', label_to_text_idx=None, accumulation_steps=1):
    """
    Run one epoch with mixed precision training, optimized label lookup, and gradient accumulation.
    
    Args:
        label_to_text_idx (dict): Precomputed mapping from CIFAR label to text embedding index
        accumulation_steps (int): Number of batches to accumulate gradients over (default: 1)
    """
    from torch.cuda.amp import autocast, GradScaler
    
    model.train() if mode == 'train' else model.eval()
    scaler = GradScaler() if mode == 'train' and device.type == 'cuda' else None
    
    total_loss = 0
    total_sim = 0
    count = 0
    
    if mode == 'train':
        optimizer.zero_grad()

    with torch.no_grad() if mode == 'eval' else torch.enable_grad():
        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc=f"[{mode.upper()}]" if mode=="train" else None)):
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            with autocast(enabled=(scaler is not None)):
                _, visual_proj = model(images)
                
                # Used precomputed mapping
                if label_to_text_idx is not None:
                    batch_text_idx = [label_to_text_idx[l.item()] for l in labels]
                else:
                    batch_text_idx = [class_words.index(label_to_word[l.item()]) for l in labels]
                
                batch_text_emb = text_emb[batch_text_idx].to(device, non_blocking=True)
                loss = compute_contrastive_loss(visual_proj, batch_text_emb, temperature,  smoothing=0.1)
                
                # GRADIENT ACCUMULATION: Scale loss by accumulation steps
                if mode == 'train':
                    loss = loss / accumulation_steps

            if mode == "train":
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                # GRADIENT ACCUMULATION: Update only after N batches
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    if scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()

            # Track metrics (unscale loss for accurate reporting)
            total_loss += (loss.item() * accumulation_steps) * len(images)
            V = F.normalize(visual_proj, p=2, dim=1)
            S = F.normalize(batch_text_emb, p=2, dim=1)
            total_sim += (V * S).sum(dim=1).sum().item()
            count += len(images)
            
    return total_loss/count, total_sim/count


def train_with_early_stopping(model, dataloaders, text_emb, class_words, label_to_word, config, device):
    """
    Train model with early stopping, warmup, gradient accumulation, and optimized lookups.
    """
    # Get accumulation steps from config
    accumulation_steps = config.get('accumulation_steps', 1)
    
    # DIFFERENT LR for backbone vs projection
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    projection_params = list(model.projection.parameters())
    
    optimizer = torch.optim.AdamW([
        {'params': backbone_params, 'lr': config['lr'] * 0.1},  # 10× lower LR for backbone
        {'params': projection_params, 'lr': config['lr']}       # Normal LR for projection
    ], weight_decay=config['weight_decay'])
    
    print(f"\n{'='*70}")
    print(f"Optimizer Configuration:")
    print(f"   Backbone LR: {config['lr'] * 0.1:.6f} ({len(backbone_params)} param groups)")
    print(f"   Projection LR: {config['lr']:.6f}")
    print(f"   Weight decay: {config['weight_decay']}")
    if accumulation_steps > 1:
        print(f"  Gradient accumulation: {accumulation_steps} steps")
        print(f"  Effective batch size: {config['batch_sizes']['train']} × {accumulation_steps} = {config['batch_sizes']['train'] * accumulation_steps}")
    else:
        print(f"  └─ Batch size: {config['batch_sizes']['train']}")
    print(f"{'='*70}")

    # Longer warmup for backbone fine-tuning
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    # warmup_epochs = 10  # Longer warmup (backbone needs gentle start) ## BEST FOR EVEN V7 and V8
    warmup_epochs = 15  # Even longer warmup for stability
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'] - warmup_epochs, eta_min=1e-7)
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # Precompute label-to-text-index mapping
    label_to_text_idx = {
        cifar_label: class_words.index(label_to_word[cifar_label])
        for cifar_label in range(100)
        if label_to_word.get(cifar_label) in class_words
    }
    
    best_val_sim, patience_counter, best_epoch = -float('inf'), 0, 0
    best_val_loss = float('inf')
    history = defaultdict(list)
    
    print(f"\n{'='*70}\nTraining (max {config['epochs']} epochs, patience={config['patience']})\n{'='*70}")
    
    try:
        for epoch in range(1, config['epochs'] + 1):
            # Pass accumulation_steps to run_epoch
            train_loss, _ = run_epoch(
                model, dataloaders['train'], text_emb, class_words, label_to_word, 
                optimizer, config['temperature'], device, 'train', label_to_text_idx, accumulation_steps
            )
            val_loss, val_sim = run_epoch(
                model, dataloaders['val'], text_emb, class_words, label_to_word, 
                None, config['temperature'], device, 'eval', label_to_text_idx, accumulation_steps=1  # No accumulation for validation
            )
            
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            
            for metric, value in zip(['train_loss', 'val_loss', 'val_similarity', 'learning_rate'], 
                                    [train_loss, val_loss, val_sim, current_lr]):
                history[metric].append(value)
            
            print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                  f"Val Sim: {val_sim:.4f} | LR: {current_lr:.6f}")
            
            if val_sim > best_val_sim:
                best_val_sim, best_val_loss, best_epoch, patience_counter = val_sim, val_loss, epoch, 0
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_loss': val_loss,
                    'val_similarity': val_sim, 'class_words': class_words, 'text_embeddings': text_emb.cpu(),
                    'history': dict(history), 'projection_head': model.projection.state_dict(),
                    'config': config,  #Save config for reproducibility
                }, config['save_path'])
                print(f"  ✓ New best model saved! (Val Sim: {val_sim:.4f})")
            else:
                patience_counter += 1
                print(f"  → No improvement ({patience_counter}/{config['patience']})")
            
            if patience_counter >= config['patience']:
                print(f"\n{'='*70}\nEarly stopping at epoch {epoch}\nBest: {best_epoch} (Val Sim: {best_val_sim:.4f})\n{'='*70}")
                break
    except Exception as e:
        print(f"\n  Training crashed with error: {str(e)}")
        print(f"Attempting to continue with best saved model...")
    
    return dict(history), best_epoch, best_val_sim, best_val_loss

# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def collect_embeddings(model, dataloader, device):
    """Collect all embeddings and labels from dataset."""
    model.eval()
    all_visual, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Collecting embeddings"):
            images = images.to(device)
            _, visual_proj = model(images)
            all_visual.append(F.normalize(visual_proj, p=2, dim=1).cpu())
            all_labels.extend(labels.tolist())
    return torch.cat(all_visual, dim=0).numpy(), all_labels


def compute_alignment_metrics(visual_emb, labels, text_emb, class_words, label_to_word):
    """Compute comprehensive alignment metrics in one pass."""
    class_sims = defaultdict(list)
    for i, label in enumerate(labels):
        if (word := label_to_word[label]) in class_words:
            sim = np.dot(visual_emb[i], text_emb[class_words.index(word)])
            class_sims[word].append(sim)
    
    stats = sorted([{
        'word': word, 'mean': np.mean(sims), 'std': np.std(sims), 
        'min': np.min(sims), 'max': np.max(sims), 'count': len(sims)
    } for word, sims in class_sims.items()], key=lambda x: x['mean'], reverse=True)
    
    sim_matrix = cosine_similarity(visual_emb, text_emb)
    i2t_recalls = {k: 0 for k in [1, 5, 10]}
    t2i_recalls = {k: 0 for k in [1, 5, 10]}
    
    for i, label in enumerate(labels):
        if (word := label_to_word[label]) in class_words:
            correct_idx = class_words.index(word)
            ranking = np.argsort(-sim_matrix[i])
            for k in i2t_recalls:
                if correct_idx in ranking[:k]: i2t_recalls[k] += 1
    
    for class_idx, word in enumerate(class_words):
        class_img_idx = [i for i, l in enumerate(labels) if label_to_word[l] == word]
        if class_img_idx:
            ranking = np.argsort(-sim_matrix[:, class_idx])
            for k in t2i_recalls:
                if any(idx in ranking[:k] for idx in class_img_idx): t2i_recalls[k] += 1
    
    return stats, i2t_recalls, t2i_recalls, sim_matrix


def print_analysis_results(stats, i2t_recalls, t2i_recalls, n_samples, n_classes):
    """Print comprehensive analysis results."""
    print("\n📊 Per-Class Similarity Analysis:")
    print("-" * 70)
    for title, data in [("Top 10 Best Aligned Classes:", stats[:10]), 
                        ("Bottom 10 Worst Aligned Classes:", stats[-10:])]:
        print(f"\n{title}")
        for i, s in enumerate(data, 1):
            print(f"{i:2d}. {s['word']:15s} | Mean: {s['mean']:.4f} ± {s['std']:.4f}")
    
    print("\n📊 Retrieval Performance:")
    print("-" * 70)
    for name, recalls, total in [("Image-to-Text", i2t_recalls, n_samples), 
                                 ("Text-to-Image", t2i_recalls, n_classes)]:
        print(f"\n{name} Retrieval (Recall@K):")
        for k, count in recalls.items():
            print(f"  Recall@{k:2d}: {count/total*100:.2f}% ({count}/{total})")

def print_example_retrievals(sim_matrix, labels, class_words, label_to_word, n_examples=5):
    """Print text-based retrieval examples."""
    print("\n📸 Example Image-to-Text Retrievals:")
    print("-" * 70)
    
    display_idx = np.random.choice(len(labels), size=n_examples, replace=False)
    
    for idx in display_idx:
        label = labels[idx]
        true_word = label_to_word[label]
        
        sims = sim_matrix[idx]
        top_5_idx = np.argsort(-sims)[:5]
        top_5_words = [class_words[i] for i in top_5_idx]
        top_5_sims = [sims[i] for i in top_5_idx]
        
        correct_sim = sims[class_words.index(true_word)]
        correct_rank = np.where(np.argsort(-sims) == class_words.index(true_word))[0][0] + 1
        
        print(f"\nTest Image #{idx}:")
        print(f"  True class: '{true_word}' (similarity: {correct_sim:.4f}, rank: {correct_rank})")
        print(f"  Top 5 predictions:")
        for rank, (word, sim) in enumerate(zip(top_5_words, top_5_sims), 1):
            marker = "✓" if word == true_word else " "
            print(f"    {rank}. {marker} {word:15s} (similarity: {sim:.4f})")

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_visualizations(sim_matrix, labels, class_words, label_to_word, test_indices, images=None, names=None, predictions=None):
    """Create all visualizations in one coordinated function."""
    if images and names and predictions:
        print(f"\n📸 Creating OOD visualization for {len(images)} images...")
        n_imgs = len(images)
        n_cols = min(4, n_imgs)
        n_rows = (n_imgs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 7*n_rows))
        axes = [axes] if n_rows == 1 and n_cols == 1 else axes.flatten()
        
        for i, (img, name, pred) in enumerate(zip(images, names, predictions)):
            axes[i].imshow(img); axes[i].axis('off')
            pred_text = f"{name.upper()}\n\nTop matches:\n"
            for rank, (word, sim) in enumerate(zip(pred['words'][:5], pred['sims'][:5]), 1):
                pred_text += f"{rank}. {word} ({sim:.3f})\n"
            axes[i].set_title(pred_text, fontsize=11, ha='center', color='darkblue', fontweight='bold', pad=12)
        
        for j in range(i+1, len(axes)): 
            axes[j].axis('off'); axes[j].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('ood_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()

    else:
        print("\n📊 Creating confusion matrix...")
        n_classes = len(class_words)
        conf_matrix = np.zeros((n_classes, n_classes))
        
        for i, label in enumerate(labels):
            if (word := label_to_word[label]) in class_words:
                true_idx = class_words.index(word)
                pred_idx = np.argmax(sim_matrix[i])
                conf_matrix[true_idx, pred_idx] += 1
        
        conf_matrix = conf_matrix / (conf_matrix.sum(axis=1, keepdims=True) + 1e-10)
        
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(conf_matrix, xticklabels=class_words, yticklabels=class_words,
                    cmap='Blues', ax=ax, cbar_kws={'label': 'Probability'}, square=True)
        ax.set_xlabel('Predicted Class'); ax.set_ylabel('True Class')
        ax.set_title('Confusion Matrix (All Classes)', fontsize=14, fontweight='bold')
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0, fontsize=9)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()

        print("\n📸 Creating retrieval examples...")
        test_raw = CIFAR100Filtered(split="val", transform=transforms.Compose([transforms.Resize(224), transforms.ToTensor()]))
        
        fig, axes = plt.subplots(3, 4, figsize=(16, 12))
        for plot_idx, ax in enumerate(axes.flatten()):
            if plot_idx >= 12: break
            ex_idx = random.randint(0, len(labels)-1)
            original_idx = test_indices[ex_idx]
            img, label = test_raw[original_idx]
            
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis('off')
            
            true_word = label_to_word[label]
            sims = sim_matrix[ex_idx]
            top_5_idx = np.argsort(-sims)[:5]
            top_5_words = [class_words[i] for i in top_5_idx]
            top_5_sims = [sims[i] for i in top_5_idx]

            pred_text = f"GT: {true_word}\n"
            for rank, (word, sim) in enumerate(zip(top_5_words, top_5_sims), 1):
                marker = "✓" if word == true_word else "✗"
                pred_text += f"{rank}. {marker} {word}: {sim:.2f}\n"

            if top_5_words[0] == true_word:
                title_color = "green"
            elif true_word in top_5_words:
                title_color = "#CC8A00"
            else:
                title_color = "red"

            ax.set_title(pred_text, fontsize=9, ha='left', fontfamily='monospace',
                        color=title_color, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('retrieval_examples.png', dpi=300, bbox_inches='tight')
        plt.show()
        

# =============================================================================
# OOD PROCESSING
# =============================================================================

def process_ood_images(model, image_urls, text_emb, class_words, device):
    """Download and process OOD images in one function."""
    print(f"\nDownloading {len(image_urls)} OOD test images...")
    images, names, headers = [], [], {'User-Agent': 'Mozilla/5.0', 'Accept': 'image/*'}
    
    for desc, url in image_urls.items():
        try:
            response = requests.get(url, timeout=30, headers=headers)
            if response.status_code == 200:
                img = Image.open(BytesIO(response.content)).convert('RGB')
                images.append(img.resize((224, 224), Image.BILINEAR))
                names.append(desc)
                print(f"  ✓ Downloaded: {desc}")
        except Exception as e:
            print(f"  ✗ Error downloading {desc}: {str(e)[:50]}")
    
    if not images: return [], [], []
    
    print(f"\n🔬 Processing {len(images)} OOD images...")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    to_tensor = transforms.ToTensor()
    
    model.eval()
    with torch.no_grad():
        ood_emb = []
        for img in images:
            img_tensor = normalize(to_tensor(img).unsqueeze(0).to(device))
            _, visual_proj = model(img_tensor)
            ood_emb.append(F.normalize(visual_proj, p=2, dim=1).cpu().numpy()[0])
    
    ood_emb = np.array(ood_emb)
    predictions = []
    for emb in ood_emb:
        sims = cosine_similarity(emb.reshape(1, -1), text_emb)[0]
        top_5_idx = np.argsort(-sims)[:5]
        predictions.append({
            'words': [class_words[j] for j in top_5_idx],
            'sims': [sims[j] for j in top_5_idx]
        })
    
    return images, names, predictions