"""
train_best_model.py
====================
Reproduces training of the best Skip-Gram model (Config F / EXP8-WideContext).

Pipeline
--------
Stage 1 -- Corpus augmentation & Skip-Gram retraining
    1. Download Visual Genome region descriptions   -> vg_text.txt
    2. Generate synthetic sentences for 68 missing CIFAR-100 words
       using an 8-tier template system & manually chosen VG anchor words
    3. Combine into one corpus                      -> vg_cifar_combined.txt
    4. Build word co-occurrence network             (process_text_network)
    5. Train Skip-Gram with Config F hyperparameters

    Config F (selected model):
        embedding_dim  = 128
        context_size   = 5          <- dominant hyperparameter
        num_negative   = 10
        learning_rate  = 0.10
        dropout        = 0.35
        weight_decay   = 5e-4
        label_smoothing= 0.10
        epochs         = 50
        batch_size     = 2048
        patience       = 6
        rare_threshold = 0.00015

    Output: best_skipgram_523words.pth

Stage 2 -- Five-phase evolutionary refinement (optional)
    Run each phase script sequentially after Stage 1:
        python phases/phase1_bert_ga.py
        python phases/phase2_hub_correction.py
        python phases/phase3_targeted_blending.py   # Config F only
        python phases/phase4_bert_guided.py         # Config F only
        python phases/phase5_orthogonal_diversify.py # Config F only

Usage
-----
    # Step 1: generate corpus (run once)
    python src/corpus_generator.py --output vg_cifar_combined.txt

    # Step 2: train
    python train_best_model.py --corpus vg_cifar_combined.txt

"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'src'))
del _sys, _os

import os
import sys
import random
import argparse
import torch
import numpy as np

from skipgram_trainer import train_embeddings
from text_network_builder import process_text_network

# -----------------------------------------------------------------------------
# TRAINING -- Config F hyperparameters
# -----------------------------------------------------------------------------

CONFIG_F = dict(
    embedding_dim   = 128,
    context_size    = 5,       # dominant hyperparameter
    num_negative    = 10,
    learning_rate   = 0.10,
    dropout         = 0.35,
    weight_decay    = 5e-4,
    label_smoothing = 0.10,
    epochs          = 50,
    batch_size      = 2048,
    patience        = 6,
    validation_fraction = 0.10,
    rare_threshold  = 0.00015, # keep top ~0.015% word frequency
)

OUTPUT_MODEL = "models/best_skipgram_523words.pth"


def train(corpus_path: str) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"Corpus file not found: {corpus_path}\n"
            "Generate it first with:\n"
            "  python src/corpus_generator.py --output vg_cifar_combined.txt"
        )
    print(f"Using corpus: {corpus_path}")

    # -- Step 3: Build text co-occurrence network -----------------------------
    print("\n" + "=" * 70)
    print("BUILDING TEXT CO-OCCURRENCE NETWORK  (rare_threshold=0.00015)")
    print("=" * 70)
    network_data = process_text_network(
        corpus_path,
        rare_threshold=CONFIG_F['rare_threshold'],
        verbose=True,
    )
    print(f"\nNetwork: {network_data['graph'].number_of_nodes():,} nodes, "
          f"{network_data['graph'].number_of_edges():,} edges")

    # -- Step 4: Train with Config F params -----------------------------------
    print("\n" + "=" * 70)
    print("TRAINING  --  Config F  (5-hop, 128-dim, 10 negatives)")
    print("=" * 70)

    result = train_embeddings(
        network_data       = network_data,
        embedding_dim      = CONFIG_F['embedding_dim'],
        batch_size         = CONFIG_F['batch_size'],
        epochs             = CONFIG_F['epochs'],
        learning_rate      = CONFIG_F['learning_rate'],
        num_negative       = CONFIG_F['num_negative'],
        validation_fraction= CONFIG_F['validation_fraction'],
        context_size       = CONFIG_F['context_size'],
        dropout            = CONFIG_F['dropout'],
        weight_decay       = CONFIG_F['weight_decay'],
        label_smoothing    = CONFIG_F['label_smoothing'],
        patience           = CONFIG_F['patience'],
        device             = None,   # auto-detect GPU/CPU
        save_plot          = True,   # saves training_loss.png
    )

    # -- Step 5: Save model ---------------------------------------------------
    save_data = {
        'nodes'           : result['nodes'],
        'embeddings'      : result['embeddings'],
        'model_state_dict': result['model'].state_dict(),
        'vocab_size'      : len(result['nodes']),
        'embedding_dim'   : CONFIG_F['embedding_dim'],
        'config'          : CONFIG_F,
    }
    os.makedirs(os.path.dirname(OUTPUT_MODEL), exist_ok=True)
    torch.save(save_data, OUTPUT_MODEL)
    print(f"\nModel saved: {OUTPUT_MODEL}")
    print(f"  Vocabulary : {len(result['nodes'])} words")
    print(f"  Embeddings : {result['embeddings'].shape}")
    print(f"\nNext steps (Stage 2 -- GA refinement):")
    print("  python refinement/phase1_bert_ga.py")
    print("  python refinement/phase2_hub_correction.py")
    print("  python refinement/phase3_targeted_blending.py")
    print("  python refinement/phase4_bert_guided.py")
    print("  python refinement/phase5_orthogonal_diversify.py")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Config-F Skip-Gram model.",
        epilog="Generate the corpus first: python src/corpus_generator.py --output vg_cifar_combined.txt"
    )
    parser.add_argument(
        "--corpus", required=True, metavar="PATH",
        help="Path to the combined corpus file produced by corpus_generator.py "
             "(e.g. vg_cifar_combined.txt)"
    )
    args = parser.parse_args()
    train(corpus_path=args.corpus)
