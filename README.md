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
|   +-- corpus_generator.py            # Generates the augmented VG+CIFAR training corpus
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
+-- image_encoder/                     # CLIP-style image encoder (standalone)
|   +-- lab8.py                        # ImageEncoder model, dataset, training loop
|   +-- train.py                       # CLI runner to retrain the image encoder
|   +-- inference.py                   # Load from HuggingFace + predict / predict_tta
|   +-- report_image_model.md          # Technical report for the image encoder
|
+-- train_best_model.py                # End-to-end training script for Config F (best model)
+-- requirements.txt                   # Python dependencies
+-- report.md                          # Full project report
```

---

## Quickstart - Reproduce All Results

### 1. Install dependencies
```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
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
# Step 1: Generate the training corpus
# Downloads Visual Genome (~400 MB, cached), generates ~408K synthetic
# sentences for the 68 missing CIFAR-100 words, saves combined corpus.
python src/corpus_generator.py --output vg_cifar_combined.txt

# Step 2: Train Config F Skip-Gram model
# Builds co-occurrence network, trains, saves models/best_skipgram_523words.pth
python train_best_model.py --corpus vg_cifar_combined.txt

# Step 3 (optional): Run 5-phase refinement pipeline
python refinement/phase1_bert_ga.py
python refinement/phase2_hub_correction.py
python refinement/phase3_targeted_blending.py
python refinement/phase4_bert_guided.py
python refinement/phase5_orthogonal_diversify.py
```

---

## Models

| File | Description |
|---|---|
| models/best_skipgram_523words.pth | Trained Skip-Gram embeddings after 5-phase refinement (1.2 MB) |
| models/best_cifar100_projection.pth | Linear projection layer for ImageNet transfer (8.7 MB) |

Both models are also published on HuggingFace:
- Skip-Gram: [`haripra1112001/visual-skipgram-cifar100`](https://huggingface.co/haripra1112001/visual-skipgram-cifar100)
- Image encoder: [`haripra1112001/clip-cifar100-mobilenet`](https://huggingface.co/haripra1112001/clip-cifar100-mobilenet)

---

## Image Encoder (`image_encoder/`)

A CLIP-style visual encoder that aligns CIFAR-100 images with the Skip-Gram text embeddings above. Achieves **77.77% Image-to-Text Recall@1** (vs 1% random baseline).

**Quick inference** (downloads checkpoint automatically):
```bash
cd image_encoder
pip install torch torchvision huggingface_hub pillow

# Single image
python inference.py --image dog.jpg

# 8-view TTA (more accurate)
python inference.py --image dog.jpg --tta --top_k 3
```

**Retrain from scratch:**
```bash
cd image_encoder
python train.py --text_emb ../models/best_skipgram_523words.pth
```

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
