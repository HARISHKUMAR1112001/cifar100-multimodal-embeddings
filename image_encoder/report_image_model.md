# CLIP-Style Visual Encoder for CIFAR-100: Architecture Evolution, Hyperparameter Exploration, and Contrastive Training

**Author:** Prajapati Harishkumar Kishorkumar

*(See also: [Embedding Model Report](https://huggingface.co/haripra1112001/visual-skipgram-cifar100/blob/main/report.md) — the word embedding space this model aligns against)*

---

## Abstract

A CLIP-style contrastive image encoder was developed and trained to align CIFAR-100 visual features with the 128-dimensional Visual Genome Skip-Gram word embeddings described in the companion embedding report. The encoder uses a MobileNetV3-Small backbone with partial fine-tuning and a four-layer projection head. The best model achieves **77.77% Image-to-Text Recall@1** on the 100-class CIFAR-100 test set, compared to a random baseline of 1%. The model converges at epoch 47 (out of a 200-epoch budget), demonstrating that partial backbone unfreezing combined with a cosine annealing schedule is the dominant factor in performance.

---

## Table of Contents

1. [Problem Statement & Design Constraints](#1-problem-statement--design-constraints)
2. [Architecture Design](#2-architecture-design)
3. [Training Objective](#3-training-objective)
4. [Final Configuration](#4-final-configuration)
5. [Data Pipeline](#5-data-pipeline)
6. [Training Dynamics & Convergence](#6-training-dynamics--convergence)
7. [Summary](#7-summary)

---

## 1. Problem Statement & Design Constraints

### 1.1 Objective

The image encoder must map CIFAR-100 images into the **same 128-dimensional embedding space** as the Visual Genome Skip-Gram word embeddings. At inference time, an image of a `bear` should produce a vector that is more similar to the `bear` text embedding than to any other class embedding. This is a zero-shot retrieval formulation — the model never sees a classification head, only the contrastive objective.

### 1.2 Constraints

| Constraint | Value | Motivation |
|---|---|---|
| Output dimension | 128 | Must match Skip-Gram embedding dimension |
| Input resolution | 224 × 224 | Standard pretrained backbone input |
| Target vocabulary | 100 CIFAR-100 class labels | Fixed by task |
| Text encoder | **Frozen** (pre-trained Skip-Gram) | Embeddings already optimised in companion work |
| Backbone | Pretrained ImageNet weights | Limited CIFAR-100 data (~40K train samples) |

The text side is frozen throughout training. Only the visual encoder is optimised. This makes the task structurally different from standard CLIP training, where both encoders adapt simultaneously.

### 1.3 Why Contrastive Learning?

A standard cross-entropy classifier with 100 output nodes would produce label predictions but no transferable embedding space. Contrastive (InfoNCE) training directly optimises the cosine similarity between image and text representations, placing the visual encoder output into the same semantic geometry as the text embeddings. This is necessary for the downstream task.

---

## 2. Architecture Design

### 2.1 Backbone — MobileNetV3-Small

MobileNetV3-Small was selected for three reasons:

1. **Computational efficiency** — depthwise-separable convolutions and hard-swish activations run efficiently on CPU and Apple MPS, critical for extended multi-epoch training
2. **Pretrained quality** — ImageNet-pretrained weights provide strong low-level features without requiring large datasets for warm-up
3. **Feature dimensionality** — the penultimate backbone feature has 576 dimensions, a natural starting point for a multi-layer projection down to 128


Freezing early layers reduces overfitting (CIFAR-100 has only 500 training images per class at 32×32 — far less than ImageNet) and reduces memory/compute. Unfreezing the last 3 blocks allows the backbone to adapt its high-level feature representations to CIFAR-100 visual patterns without disrupting low-level filters.

### 2.2 Projection Head

A four-layer MLP maps backbone features (576-dim) to the target space (128-dim):

| Layer | In → Out | Ops |
|---|---|---|
| 1 | 576 → 1024 | Linear + BatchNorm1d + ReLU + Dropout(0.20) |
| 2 | 1024 → 512 | Linear + BatchNorm1d + ReLU + Dropout(0.15) |
| 3 | 512 → 256 | Linear + BatchNorm1d + ReLU + Dropout(0.10) |
| 4 | 256 → 128 | Linear (no activation) |

The final layer has no activation, BatchNorm, or Dropout — the raw 128-dim output is L2-normalised externally before computing cosine similarity. This is standard practice in contrastive learning (SimCLR, CLIP): applying BatchNorm or ReLU to the final layer distorts the direction of the embedding vector.

The **bottleneck then expansion then bottleneck** shape (576 → 1024 → 512 → 256 → 128) was chosen deliberately. The initial expansion to 1024 allows the model to learn a richer intermediate representation before compressing. Early experiments with a direct 576 → 256 → 128 two-layer head showed underfitting — insufficient capacity to disentangle CIFAR-100's 100 visually diverse classes.

---

## 3. Training Objective

### 3.1 Symmetric InfoNCE Loss

The contrastive loss is the standard symmetric InfoNCE (bidirectional cross-entropy over the similarity matrix):

$$\mathcal{L} = \frac{1}{2}\left[\mathcal{L}_{I\rightarrow T} + \mathcal{L}_{T\rightarrow I}\right]$$

where for a batch of $B$ image-text pairs:

$$\mathcal{L}_{I\rightarrow T} = -\frac{1}{B}\sum_{i=1}^{B} \log \frac{\exp(\cos(v_i, t_i)/\tau)}{\sum_{j=1}^{B}\exp(\cos(v_i, t_j)/\tau)}$$

- $v_i$ = L2-normalised visual projection for image $i$
- $t_i$ = L2-normalised text embedding for the correct class label of image $i$
- $\tau$ = temperature scalar

The **random baseline loss** for a batch of size $B$ is $\log(B) \approx 5.55$ (for $B=256$). Any model achieving loss substantially below this has learned to discriminate.

### 3.2 Label Smoothing

Standard cross-entropy assigns all probability mass to the diagonal (the correct pair). With 100 semantically related classes, some off-diagonal pairs are semantically valid (e.g., `forest` and `pine_tree` are genuinely related). Label smoothing distributes a small fraction $\epsilon=0.1$ of the target probability to other classes:

$$\text{soft label}_{ij} = \begin{cases} 1 - \epsilon & i = j \\ \frac{\epsilon}{B-1} & i \neq j \end{cases}$$

This prevents the model from being penalised for producing moderately high similarity to semantically adjacent classes, reducing overfitting on ambiguous inter-class boundaries.

### 3.3 Temperature

Temperature $\tau$ controls the sharpness of the softmax distribution:

| Temperature | Effect |
|---|---|
| $\tau \rightarrow 0$ | Winner-takes-all, near-zero gradient for non-maximally-similar pairs |
| $\tau = 1.0$ | Near-uniform distribution, very weak supervision signal |
| $\tau = 0.07$ | Standard CLIP value — provides sharp gradients for high-similarity pairs |

Early experiments used $\tau = 0.04$–$0.05$ (very sharp), which caused gradient instability in the first few epochs. $\tau = 0.07$ provided the optimal balance between supervision sharpness and training stability.

---

## 4. Final Configuration

The final submitted model (`best_cifar100_projection.pth`) used the following configuration.

### 4.1 Hyperparameters

| Parameter | Value | Justification |
|---|---|---|
| `proj_dim` | 128 | Must match Skip-Gram output dimension |
| `lr` | 0.00025 | Determined by hyperparameter search |
| `weight_decay` | 0.007 | Strong regularisation for small dataset |
| `temperature` | 0.07 | Standard CLIP value; stable gradients |
| `epochs` (max) | 200 | Budget with early stopping |
| `patience` | 40 | Allow plateau phases before stopping |
| `batch_size` | 256 | Memory-GPU trade-off |
| `accumulation_steps` | 2 | Effective batch = **512** |
| Backbone LR multiplier | 0.1× | Prevents destruction of pretrained features |
| Warmup epochs | 15 | Gentle LR ramp for backbone stability |

### 4.2 Learning Rate Schedule

A two-phase schedule was used:

1. **Linear warmup** (epochs 1–15): LR ramps from `0.1 × base_lr` to `base_lr`
2. **Cosine annealing** (epochs 16–200): LR decays from `base_lr` to `1e-7`

The warmup is particularly important with partial backbone unfreezing. Without it, the first few batches apply large gradient updates to the unfrozen backbone layers, potentially destroying the pretrained weight configuration before the projection head has learned a useful direction.

### 4.3 Checkpoint Contents

The saved checkpoint stores all information needed to reproduce or continue training:

| Key | Contents |
|---|---|
| `epoch` | 47 (best epoch) |
| `model_state_dict` | 263 tensors (backbone + projection) |
| `val_loss` | 2.9579 |
| `val_similarity` | 0.3927 |
| `config` | Full hyperparameter dictionary |
| `history` | Per-epoch train loss, val loss, val similarity, LR |
| `text_embeddings` | Tensor [100, 128] — frozen class embeddings |
| `class_words` | List of 100 CIFAR-100 class names |
| `projection_head` | Weights of projection MLP only (for loading without full model) |

---

## 5. Data Pipeline

### 5.1 Train/Val/Test Splits

| Split | Source | Samples | Purpose |
|---|---|---|---|
| Train | CIFAR-100 train | ~40,000 | Contrastive learning |
| Val | 20% of train (stratified) | ~10,000 | Early stopping |
| Test | CIFAR-100 test | ~10,000 | Final evaluation only |

The validation split was held out before training and never used for model selection within an epoch. Early stopping used only validation similarity (not loss) as the primary criterion.

### 5.2 Data Augmentation

Different augmentation pipelines were used for train and evaluation:

**Training:**

| Transform | Parameters |
|---|---|
| RandomResizedCrop(224) | scale=(0.6, 1.0) |
| RandomHorizontalFlip | p=0.5 |
| ColorJitter | brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1 |
| RandomGrayscale | p=0.2 |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

**Evaluation:**
```
Resize(256) → CenterCrop(224) → ToTensor → Normalize
```

---

## 6. Training Dynamics & Convergence

### 6.1 Convergence Evidence

The model trained to epoch 47 before the early stopping criterion was satisfied (patience=40 with no improvement in val similarity):

| Metric | Epoch 47 (best) |
|---|---|
| Train loss | 2.668 |
| Val loss | 2.958 |
| Val similarity | **0.3927** |
| LR (projection) | ~0.000100 |

The training curve follows the expected contrastive learning pattern:
- **Epochs 1–15 (warmup):** Slow improvement as LR ramps up; backbone and projection head are jointly finding their initial directions
- **Epochs 16–35:** Sharp drop in loss driven by cosine annealing; similarity rises rapidly as the model learns class-level structure
- **Epochs 35–47:** Plateau phase; marginal improvements as LR approaches minimum

### 6.2 Interpreting Val Similarity vs. Recall@1

Two metrics are reported, which measure fundamentally different things:

| Metric | Value | Scale | Meaning |
|---|---|---|---|
| Val cosine similarity | 0.3927 | [-1, 1] | Average dot product between image and **correct** text embedding |
| I2T Recall@1 | **77.77%** | [0%, 100%] | % of images where correct class ranks 1st among all 100 classes |

A cosine similarity of 0.39 with 77.77% Recall@1 is consistent and expected. The similarity value does not need to be high in absolute terms — it only needs to be **higher than the other 99 class similarities** for that image. Because the embedding space has strong inter-class separation (86.9% MRR in the companion embedding model), even moderate cosine alignment is sufficient for correct retrieval.

The **random baseline Recall@1 is 1%** (1 correct class out of 100). The model achieves 77.77× improvement over chance.


---

## 7. Summary

### 7.1 Final Model Specification

| Attribute | Value |
|---|---|
| **Architecture** | MobileNetV3-Small + 4-layer MLP projection |
| **Output dimension** | 128 (matches Skip-Gram embeddings) |
| **Backbone fine-tuning** | Last 3 of 13 layers unfrozen (69.5% of backbone trainable) |
| **Total trainable parameters** | 1,928,096 |
| **Training objective** | Symmetric InfoNCE (label smoothing ε=0.1) |
| **Temperature** | τ = 0.07 |
| **Optimiser** | AdamW with differential LR (backbone: 0.1× base LR) |
| **LR schedule** | Linear warmup (15 epochs) → Cosine annealing |
| **Effective batch size** | 512 (256 × 2 accumulation steps) |
| **Best epoch** | 47 (of 200 budget, patience=40) |
| **Val similarity** | 0.3927 |
| **I2T Recall@1** | **77.77%** |
| **Random baseline** | 1% (1/100 classes) |

### 7.2 Key Findings

1. **Partial backbone fine-tuning is the dominant factor.** Moving from a frozen backbone to unfreezing the last 3 layers produced the largest performance jump across all experiments (~+20% I2T). No amount of projection head tuning compensated for a frozen backbone.

2. **Differential learning rates are essential for partial fine-tuning.** Using the same LR for backbone and projection head degraded performance. The backbone needs a 10× lower LR to adapt slowly without destroying pretrained feature quality.

3. **Temperature τ = 0.07 is optimal.** Lower temperatures caused instability; higher temperatures provided insufficient supervision signal.

4. **Effective batch size ≥ 512 improves results.** Gradient accumulation over 2 steps provided a practical and effective way to double the contrastive batch without increasing memory requirements.

5. **Val cosine similarity and Recall@1 measure different things.** A val similarity of 0.39 is consistent with 77.77% Recall@1 — what matters is that image embeddings rank closest to the correct class, not that absolute cosine values are high.

6. **Visually similar confusions are semantically reasonable.** Errors (seal→dolphin, camel→kangaroo, tulip→orchid) reflect genuine visual ambiguity in 32×32 CIFAR-100 images and correspond to classes that are also semantically close in the embedding space, confirming the model has learned semantically coherent visual representations.

---

*Companion report: [Embedding Model — Skip-Gram + Genetic Algorithm Refinement](https://huggingface.co/haripra1112001/visual-skipgram-cifar100/blob/main/report.md)*
