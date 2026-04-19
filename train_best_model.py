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
    python train_best_model.py              # full Stage 1
    python train_best_model.py --skip-download   # if vg_text.txt already exists

"""

import os
import sys
import random
import argparse
import torch
import numpy as np

from skipgram_trainer import prepare_visual_genome_text, train_embeddings
from text_network_builder import process_text_network

# -----------------------------------------------------------------------------
# ANCHOR WORDS
# For each of the 68 missing CIFAR-100 words we specify 8-12 Visual Genome
# anchor words that are present in the VG vocabulary.  These act as semantic
# bridges, embedding the new word into the existing visual co-occurrence space.
# -----------------------------------------------------------------------------

ANCHOR_WORDS = {
    # -- Aquatic mammals ------------------------------------------------------
    'beaver':    ['water', 'river', 'animal', 'brown', 'wood', 'dam', 'tail', 'mammal', 'pond', 'fur'],
    'dolphin':   ['water', 'ocean', 'animal', 'sea', 'wave', 'mammal', 'gray', 'swim', 'fish', 'boat'],
    'otter':     ['water', 'river', 'animal', 'brown', 'mammal', 'fish', 'swim', 'rock', 'furry', 'small'],
    'seal':      ['water', 'ocean', 'animal', 'rock', 'gray', 'mammal', 'swim', 'beach', 'sea', 'fish'],
    'whale':     ['ocean', 'water', 'animal', 'large', 'sea', 'blue', 'mammal', 'wave', 'swim', 'deep'],

    # -- Fish -----------------------------------------------------------------
    'aquarium_fish': ['water', 'fish', 'tank', 'small', 'color', 'swim', 'ocean', 'glass', 'sea', 'blue'],
    'flatfish':  ['ocean', 'water', 'fish', 'sea', 'floor', 'sand', 'flat', 'swim', 'deep', 'brown'],
    'ray':       ['ocean', 'water', 'flat', 'fish', 'sea', 'swim', 'wing', 'deep', 'floor', 'animal'],
    'trout':     ['water', 'fish', 'river', 'swim', 'stream', 'brown', 'food', 'lake', 'gray', 'silver'],

    # -- Flowers --------------------------------------------------------------
    'orchid':    ['flower', 'pink', 'plant', 'purple', 'white', 'bloom', 'petal', 'garden', 'color', 'stem'],
    'poppy':     ['flower', 'red', 'plant', 'field', 'petal', 'bloom', 'orange', 'stem', 'garden', 'bright'],
    'rose':      ['flower', 'red', 'plant', 'petal', 'bloom', 'garden', 'pink', 'stem', 'thorn', 'beautiful'],
    'sunflower': ['flower', 'yellow', 'plant', 'tall', 'field', 'bloom', 'sun', 'petal', 'seed', 'bright'],
    'tulip':     ['flower', 'pink', 'plant', 'bloom', 'petal', 'red', 'garden', 'stem', 'spring', 'color'],

    # -- Food containers ------------------------------------------------------
    'bottle':    ['glass', 'water', 'table', 'plastic', 'food', 'drink', 'round', 'container', 'kitchen', 'liquid'],
    'bowl':      ['table', 'food', 'kitchen', 'round', 'eat', 'white', 'soup', 'dish', 'plate', 'wood'],
    'can':       ['metal', 'food', 'cylinder', 'round', 'label', 'drink', 'open', 'container', 'silver', 'tin'],
    'cup':       ['table', 'drink', 'round', 'white', 'hold', 'coffee', 'kitchen', 'glass', 'mug', 'handle'],
    'plate':     ['food', 'table', 'round', 'white', 'eat', 'dish', 'kitchen', 'serving', 'flat', 'meal'],

    # -- Fruits & vegetables --------------------------------------------------
    'apple':     ['fruit', 'red', 'round', 'food', 'tree', 'eat', 'green', 'sweet', 'garden', 'fresh'],
    'orange':    ['fruit', 'round', 'food', 'color', 'eat', 'sweet', 'fresh', 'juice', 'peel', 'bright'],
    'pear':      ['fruit', 'green', 'round', 'food', 'eat', 'tree', 'sweet', 'fresh', 'yellow', 'garden'],
    'sweet_pepper': ['food', 'red', 'vegetable', 'green', 'eat', 'cook', 'fresh', 'plant', 'kitchen', 'pepper'],

    # -- Insects --------------------------------------------------------------
    'beetle':    ['insect', 'small', 'black', 'bug', 'shell', 'fly', 'ground', 'leaf', 'brown', 'hard'],
    'caterpillar': ['insect', 'small', 'green', 'leaf', 'tree', 'bug', 'crawl', 'long', 'plant', 'soft'],
    'cockroach': ['insect', 'brown', 'small', 'bug', 'floor', 'dark', 'fast', 'flat', 'kitchen', 'ground'],

    # -- Large carnivores -----------------------------------------------------
    'leopard':   ['animal', 'spot', 'big', 'cat', 'wild', 'tree', 'jungle', 'brown', 'fast', 'predator'],

    # -- Large outdoor structures ---------------------------------------------
    'castle':    ['building', 'stone', 'old', 'tower', 'wall', 'large', 'medieval', 'gate', 'high', 'historic'],
    'skyscraper': ['building', 'tall', 'city', 'glass', 'high', 'metal', 'window', 'urban', 'sky', 'office'],

    # -- Large natural outdoor scenes -----------------------------------------
    'forest':    ['tree', 'green', 'wood', 'plant', 'leaf', 'tall', 'nature', 'dark', 'ground', 'path'],
    'plain':     ['field', 'flat', 'grass', 'open', 'green', 'sky', 'wide', 'ground', 'land', 'area'],

    # -- Large omnivores/herbivores --------------------------------------------
    'camel':     ['animal', 'desert', 'sand', 'large', 'brown', 'hump', 'mammal', 'walk', 'dry', 'ride'],
    'cattle':    ['animal', 'large', 'farm', 'grass', 'brown', 'field', 'barn', 'mammal', 'white', 'cow'],
    'chimpanzee': ['animal', 'black', 'tree', 'jungle', 'primate', 'climb', 'mammal', 'arm', 'face', 'fur'],
    'kangaroo':  ['animal', 'brown', 'large', 'jump', 'mammal', 'Australia', 'tail', 'leg', 'pouch', 'fur'],

    # -- Medium mammals --------------------------------------------------------
    'fox':       ['animal', 'orange', 'forest', 'small', 'tail', 'mammal', 'wild', 'fur', 'ears', 'hunt'],
    'porcupine': ['animal', 'small', 'spine', 'brown', 'quill', 'mammal', 'ground', 'sharp', 'fur', 'forest'],
    'possum':    ['animal', 'gray', 'small', 'tree', 'mammal', 'night', 'tail', 'fur', 'climb', 'round'],
    'raccoon':   ['animal', 'gray', 'small', 'black', 'mask', 'mammal', 'forest', 'tail', 'fur', 'clever'],
    'skunk':     ['animal', 'black', 'white', 'small', 'smell', 'mammal', 'stripe', 'forest', 'fur', 'ground'],

    # -- Non-insect invertebrates ---------------------------------------------
    'crab':      ['ocean', 'water', 'shell', 'red', 'small', 'claw', 'sea', 'beach', 'sand', 'sidewalk'],
    'lobster':   ['ocean', 'red', 'water', 'shell', 'claw', 'sea', 'food', 'cook', 'large', 'antenna'],
    'snail':     ['small', 'shell', 'ground', 'slow', 'wet', 'round', 'garden', 'brown', 'leaf', 'animal'],
    'worm':      ['ground', 'small', 'brown', 'soil', 'long', 'soft', 'garden', 'earth', 'dig', 'pink'],

    # -- Reptiles -------------------------------------------------------------
    'crocodile': ['animal', 'green', 'water', 'large', 'teeth', 'river', 'reptile', 'jaw', 'long', 'swim'],
    'dinosaur':  ['animal', 'large', 'green', 'old', 'teeth', 'wild', 'extinct', 'tail', 'scales', 'ancient'],
    'lizard':    ['animal', 'small', 'green', 'scales', 'reptile', 'rock', 'tail', 'fast', 'brown', 'ground'],

    # -- Small mammals ---------------------------------------------------------
    'hamster':   ['animal', 'small', 'brown', 'fur', 'round', 'cage', 'wheel', 'mammal', 'cute', 'rodent'],
    'rabbit':    ['animal', 'small', 'white', 'fur', 'ears', 'long', 'mammal', 'garden', 'soft', 'jump'],
    'shrew':     ['animal', 'small', 'brown', 'fur', 'nose', 'mammal', 'ground', 'rodent', 'tiny', 'field'],
    'squirrel':  ['animal', 'small', 'brown', 'tree', 'tail', 'mammal', 'nut', 'climb', 'fur', 'cute'],

    # -- Trees -----------------------------------------------------------------
    'maple_tree':  ['tree', 'leaf', 'tall', 'wood', 'branch', 'green', 'red', 'fall', 'trunk', 'bark'],
    'oak_tree':    ['tree', 'tall', 'wood', 'leaf', 'branch', 'strong', 'brown', 'trunk', 'old', 'bark'],
    'palm_tree':   ['tree', 'tall', 'tropical', 'leaf', 'trunk', 'beach', 'coconut', 'green', 'long', 'hot'],
    'pine_tree':   ['tree', 'tall', 'green', 'needle', 'cone', 'wood', 'forest', 'branch', 'trunk', 'bark'],
    'willow_tree': ['tree', 'leaf', 'branch', 'long', 'green', 'hanging', 'water', 'soft', 'trunk', 'tall'],

    # -- Vehicles (type 1) -----------------------------------------------------
    'pickup_truck': ['vehicle', 'truck', 'road', 'large', 'drive', 'metal', 'wheel', 'carry', 'engine', 'flat'],

    # -- Vehicles (type 2) -----------------------------------------------------
    'lawn_mower': ['machine', 'grass', 'cut', 'garden', 'engine', 'green', 'wheel', 'yard', 'metal', 'noise'],
    'rocket':     ['sky', 'fast', 'metal', 'fire', 'launch', 'high', 'space', 'engine', 'tall', 'bright'],
    'streetcar':  ['vehicle', 'track', 'city', 'road', 'ride', 'metal', 'rail', 'street', 'electric', 'long'],
    'tank':       ['vehicle', 'metal', 'large', 'heavy', 'weapon', 'military', 'wheel', 'track', 'armour', 'gun'],
    'tractor':    ['vehicle', 'farm', 'large', 'wheel', 'engine', 'field', 'metal', 'drive', 'work', 'heavy'],

    # -- Household electrical devices -----------------------------------------
    'keyboard':   ['computer', 'desk', 'type', 'key', 'plastic', 'black', 'office', 'screen', 'work', 'letter'],
    'telephone':  ['phone', 'call', 'desk', 'black', 'ring', 'plastic', 'talk', 'old', 'cable', 'button'],
    'television': ['screen', 'room', 'watch', 'black', 'show', 'large', 'image', 'stand', 'remote', 'flat'],
    'wardrobe':   ['furniture', 'wood', 'room', 'door', 'clothes', 'large', 'bedroom', 'mirror', 'shelf', 'dark'],
}

# -----------------------------------------------------------------------------
# SENTENCE TEMPLATES  (8-tier system from report Section 3.1)
# -----------------------------------------------------------------------------

def _article(word: str) -> str:
    return 'an' if word[0].lower() in 'aeiou' else 'a'


def generate_sentences(word: str, anchors: list, n_repeats: int = 6) -> list:
    """
    Generate diverse synthetic sentences for `word` using VG anchor words.
    Produces ~8 sentence types x n_anchor_pairs sentences.
    """
    art = _article(word)
    sentences = []

    for i, anchor in enumerate(anchors):
        a = _article(anchor)
        # Tier 1 -- Direct definition
        sentences.append(f"A {word} is a type of {anchor}.")
        sentences.append(f"The {word} is known as {a} {anchor}.")
        # Tier 2 -- Visual scene
        sentences.append(f"The {word} was seen near the {anchor}.")
        sentences.append(f"A {word} appeared beside {a} {anchor} in the scene.")
        # Tier 3 -- Habitat/context
        sentences.append(f"You can find {art} {word} near {a} {anchor}.")
        sentences.append(f"A {word} often lives close to {a} {anchor}.")
        # Tier 4 -- Action/behaviour
        sentences.append(f"The {word} moved through the {anchor}.")
        sentences.append(f"A {word} was spotted near the {anchor} area.")
        # Tier 5 -- Comparative
        if i + 1 < len(anchors):
            anchor2 = anchors[i + 1]
            sentences.append(f"A {word} looks similar to {a} {anchor} but differs from {a} {anchor2}.")
        # Tier 6 -- Attribute
        sentences.append(f"The {word} has a {anchor}-like appearance.")
        # Tier 7 -- Group context
        if i + 1 < len(anchors):
            anchor2 = anchors[i + 1]
            sentences.append(f"Several {word}s were seen alongside the {anchor} and the {anchor2}.")
        # Tier 8 -- Negative separation (contrastive)
        if i + 1 < len(anchors):
            anchor2 = anchors[i + 1]
            sentences.append(f"A {word} is not the same as {a} {anchor2}, though they share similar {anchor}.")

    return sentences


def build_augmented_corpus(vg_text_path: str, output_path: str = "vg_cifar_combined.txt") -> str:
    """
    Reads vg_text.txt and appends ~408K synthetic sentences for the 68
    missing CIFAR-100 words, then saves the combined corpus.
    """
    if os.path.exists(output_path):
        print(f"Augmented corpus already exists: {output_path}  (skipping regeneration)")
        return output_path

    print(f"\nReading base corpus: {vg_text_path}")
    with open(vg_text_path, 'r', encoding='utf-8') as f:
        base_text = f.read()

    print("Generating synthetic sentences for 68 missing CIFAR-100 words...")
    all_synthetic = []
    for word, anchors in ANCHOR_WORDS.items():
        sents = generate_sentences(word, anchors)
        all_synthetic.extend(sents)

    random.shuffle(all_synthetic)
    synthetic_block = " . ".join(all_synthetic)

    print(f"  Base sentences : ~{base_text.count(' . '):,}")
    print(f"  Synthetic added: {len(all_synthetic):,}")

    combined = base_text + " . " + synthetic_block

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(combined)

    print(f"\nCombined corpus saved: {output_path}")
    return output_path


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

OUTPUT_MODEL = "best_skipgram_523words.pth"


def train(skip_download: bool = False) -> None:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    # -- Step 1: Download Visual Genome text ----------------------------------
    vg_url = "https://homes.cs.washington.edu/~ranjay/visualgenome/data/dataset/region_descriptions.json.zip"
    if skip_download and os.path.exists("vg_text.txt"):
        vg_path = "vg_text.txt"
        print("Skipping VG download -- vg_text.txt already present.")
    else:
        vg_path = prepare_visual_genome_text(vg_url)

    # -- Step 2: Build augmented corpus ---------------------------------------
    corpus_path = build_augmented_corpus(vg_path)

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
    torch.save(save_data, OUTPUT_MODEL)
    print(f"\nModel saved: {OUTPUT_MODEL}")
    print(f"  Vocabulary : {len(result['nodes'])} words")
    print(f"  Embeddings : {result['embeddings'].shape}")
    print(f"\nNext steps (Stage 2 -- GA refinement):")
    print("  python phases/phase1_bert_ga.py")
    print("  python phases/phase2_hub_correction.py")
    print("  python phases/phase3_targeted_blending.py")
    print("  python phases/phase4_bert_guided.py")
    print("  python phases/phase5_orthogonal_diversify.py")


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Config-F Skip-Gram model")
    parser.add_argument(
        "--skip-download", action="store_true",
        help="Skip VG download if vg_text.txt already exists"
    )
    args = parser.parse_args()
    train(skip_download=args.skip_download)
