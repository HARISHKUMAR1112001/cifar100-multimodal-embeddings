# Extending Skip-Gram Word Embeddings to CIFAR-100

Two-stage approach: synthetic corpus augmentation + BERT-guided genetic algorithm
refinement to extend visual-grounding Skip-Gram embeddings to all 100 CIFAR-100 class labels.

**Final model:** 86.9% MRR | 93/100 perfect-clustering words | 6% contamination | 77.71% ImageNet transfer accuracy

---

## Repository Structure

```
.
+-- src/                               # Core model source code
|   +-- skipgram_embeddings.py         # Entry point: build_my_embeddings() -> vocab + embeddings
|   +-- skipgram_trainer.py            # SkipGramModel, SkipGramDataset, train_embeddings()
|   +-- text_network_builder.py        # Text corpus -> co-occurrence graph -> training data
|   +-- image_encoder.py               # ImageEncoder for ImageNet transfer evaluation
|
+-- refinement/                        # 5-phase geometric embedding refinement pipeline
|   +-- phase1_bert_ga.py              # Phase 1: BERT-guided Genetic Algorithm
|   +-- phase2_hub_correction.py       # Phase 2: Hub word detection & repulsion
|   +-- phase3_targeted_blending.py    # Phase 3: Surgical centroid blending
|   +-- phase4_bert_guided.py          # Phase 4: BERT-guided directional fix
|   +-- phase5_orthogonal_diversify.py # Phase 5: Orthogonal diversification
|
+-- evaluation/                        # Evaluation and reporting scripts
|   +-- eval_rank_mrr.py               # MRR, Precision@K, Recall@K, Hit Rate@K
|   +-- eval_cosine_threshold.py       # Cosine threshold evaluation, contamination check
|   +-- reproduce_results.py           # Reproduces ALL paper metrics + t-SNE plot
|   +-- compare_baselines.py           # Compares vs GloVe 100d/300d + fastText 300d
|
+-- models/                            # Saved model checkpoints
|   +-- best_skipgram_523words.pth     # Trained Skip-Gram embeddings (523-word vocab, 128-dim)
|   +-- best_cifar100_projection.pth   # Trained ImageNet projection layer
|
+-- outputs/                           # Generated plots (created on first run)
|   +-- tsne_cifar100.png
|   +-- baseline_comparison.png
|   +-- baseline_comparison_heatmap.png
|
+-- train_best_model.py                # End-to-end training script for Config F (best model)
+-- requirements.txt                   # Python dependencies
+-- report.md                          # Full project report
```

---

## Quickstart - Reproduce All Results

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run full evaluation (reproduces all paper metrics + t-SNE)
```bash
python evaluation/reproduce_results.py
```

Output:
- Overall MRR (86.9%)
- Excellent superclasses: 16/20
- Perfect-clustering words: 93/100
- Contamination rate: 6%
- Mean similarity, CV, top-5 avg similarity
- Per-class MRR for all 20 superclasses
- Saves outputs/tsne_cifar100.png

### Run individual evaluations
```bash
python evaluation/eval_rank_mrr.py          # MRR, Precision@K, Recall@K, Hit Rate@K
python evaluation/eval_cosine_threshold.py  # Cosine threshold, contamination check
python evaluation/compare_baselines.py      # Comparison vs GloVe and fastText baselines
```

### Train from scratch
```bash
python train_best_model.py                  # Trains Config F (5-hop, lr=0.10, dropout=0.35)
python train_best_model.py --skip-download  # Skip Visual Genome download if already present
```

---

## Models

| File | Description |
|---|---|
| models/best_skipgram_523words.pth | Trained Skip-Gram embeddings after 5-phase refinement (1.2 MB) |
| models/best_cifar100_projection.pth | Linear projection layer for ImageNet transfer (8.7 MB) |

---

## Best Model Configuration (Config F)

| Hyperparameter | Value |
|---|---|
| context_size | 5 |
| num_negative | 10 |
| learning_rate | 0.10 |
| dropout | 0.35 |
| weight_decay | 5e-4 |
| label_smoothing | 0.10 |
| epochs | 50 |
| batch_size | 2048 |
| embedding_dim | 128 |

---

## Refinement Pipeline (Phases)

Phases run sequentially on the trained Skip-Gram model. Each script is standalone.

| Phase | Script | What it does |
|---|---|---|
| 1 | refinement/phase1_bert_ga.py | BERT-guided GA moves contaminated words toward correct superclass centroid |
| 2 | refinement/phase2_hub_correction.py | Detects hub words and applies repulsion to break false clusters |
| 3 | refinement/phase3_targeted_blending.py | Surgical centroid blending for household/electrical confusion |
| 4 | refinement/phase4_bert_guided.py | BERT-guided directional fix for final 8 stubborn words |
| 5 | refinement/phase5_orthogonal_diversify.py | PCA-based orthogonal diversification for collapsed natural scene vectors |
