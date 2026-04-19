"""
==============
Computes EXACTLY the metrics reported in question_5.tex (Part B).

Reported metrics:
  1. MRR                    : 86.9%         (rank-based, all 100 words)
  2. Excellent superclasses : 16/20          (rank-based, MRR >= 0.8 per class)
  3. Perfect words          : 93/100  93.0%  (contamination check)
  4. Contamination rate     : 6/100    6.0%
  5. Semantically valid     : 1/100    1.0%
  6. Mean similarity        : 0.539          (full vocab sample)
  7. CV                     : ~23-24%        (std/mean * 100)
  8. Top-5 avg neighbor sim : 0.814          (CIFAR-100 words only)

"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
del _sys, _os

import numpy as np
from collections import defaultdict
from skipgram_embeddings import build_my_embeddings
from skipgram_trainer import find_similar_words

# -----------------------------------------------------------------------------
# CIFAR-100 SUPERCLASSES
# -----------------------------------------------------------------------------

SUPERCLASSES = {
    'aquatic_mammals':              ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish':                         ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers':                      ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers':              ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruits_vegetables':            ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices': ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture':          ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects':                      ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores':             ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man_made_outdoor_things':['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes': ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_herbivores':   ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals':               ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non_insect_invertebrates':     ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people':                       ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles':                     ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals':                ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees':                        ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1':                   ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2':                   ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

# Semantic cross-class allowances (for contamination check)
SEMANTIC_ALLOWANCES = {
    'forest':      {'allowed_classes': ['trees'],                                    'reason': 'Trees are IN forests'},
    'mountain':    {'allowed_classes': ['trees'],                                    'reason': 'Trees grow on mountains'},
    'plain':       {'allowed_classes': ['trees'],                                    'reason': 'Trees grow on plains'},
    'sea':         {'allowed_classes': ['aquatic_mammals', 'fish', 'trees'],         'reason': 'Marine life + coastal trees'},
    'cloud':       {'allowed_classes': ['trees'],                                    'reason': 'Clouds over forests'},
    'maple_tree':  {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Trees grow IN landscapes'},
    'oak_tree':    {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Trees grow IN landscapes'},
    'willow_tree': {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Willows near water'},
    'palm_tree':   {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Palms on beaches/plains'},
    'pine_tree':   {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Pines in forests/mountains'},
    'beaver':      {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Beavers live near water'},
    'dolphin':     {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Dolphins in seas'},
    'otter':       {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Otters in seas/rivers'},
    'seal':        {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Seals in seas'},
    'whale':       {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Whales in seas'},
    'crab':        {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Crabs near seas'},
    'lobster':     {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Lobsters in seas'},
    'spider':      {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Spiders in forests'},
    'worm':        {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Worms in soil/forests'},
    'snail':       {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Snails in forests'},
    'baby':        {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'boy':         {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'girl':        {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'man':         {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'woman':       {'allowed_classes': ['household_furniture', 'household_electrical_devices'], 'reason': 'People use household items'},
    'bicycle':     {'allowed_classes': ['large_man_made_outdoor_things'],            'reason': 'Bicycles on roads/bridges'},
    'bus':         {'allowed_classes': ['large_man_made_outdoor_things'],            'reason': 'Buses on roads'},
    'motorcycle':  {'allowed_classes': ['large_man_made_outdoor_things'],            'reason': 'Motorcycles on roads'},
    'pickup_truck':{'allowed_classes': ['large_man_made_outdoor_things', 'vehicles_2'], 'reason': 'Trucks on roads, semantically a vehicle'},
    'train':       {'allowed_classes': ['large_man_made_outdoor_things'],            'reason': 'Trains on infrastructure'},
    'bear':        {'allowed_classes': ['large_natural_outdoor_scenes', 'trees'],    'reason': 'Bears in forests'},
    'leopard':     {'allowed_classes': ['large_natural_outdoor_scenes', 'trees'],    'reason': 'Leopards in wild'},
    'lion':        {'allowed_classes': ['large_natural_outdoor_scenes'],             'reason': 'Lions in plains'},
    'tiger':       {'allowed_classes': ['large_natural_outdoor_scenes', 'trees'],    'reason': 'Tigers in jungles'},
    'wolf':        {'allowed_classes': ['large_natural_outdoor_scenes', 'trees'],    'reason': 'Wolves in forests'},
}

# -----------------------------------------------------------------------------
# LOAD EMBEDDINGS
# -----------------------------------------------------------------------------

vocab_dict, embeddings = build_my_embeddings()
vocab = sorted(vocab_dict.keys(), key=lambda w: vocab_dict[w])

word_to_class = {}
for superclass, members in SUPERCLASSES.items():
    for word in members:
        word_to_class[word] = superclass

ALL_CIFAR100_WORDS = [w for members in SUPERCLASSES.values() for w in members]

# -----------------------------------------------------------------------------
# METRIC 1 & 2: MRR + EXCELLENT SUPERCLASSES (rank-based, MRR >= 0.8)
# -----------------------------------------------------------------------------

def compute_mrr_for_word(word, vocab, embeddings, word_to_class):
    """MRR: reciprocal rank of first same-class neighbor."""
    if word not in vocab:
        return 0.0
    true_class = word_to_class[word]
    same_class = [w for w in SUPERCLASSES[true_class] if w != word]
    neighbors = find_similar_words(word, vocab, embeddings, top_k=50)
    for rank, (neighbor, _) in enumerate(neighbors, 1):
        if neighbor in same_class:
            return 1.0 / rank
    return 0.0

word_mrrs = {}
for word in ALL_CIFAR100_WORDS:
    word_mrrs[word] = compute_mrr_for_word(word, vocab, embeddings, word_to_class)

overall_mrr = np.mean(list(word_mrrs.values()))

# Per-class MRR
class_mrrs = {}
for superclass, members in SUPERCLASSES.items():
    valid = [w for w in members if w in vocab]
    if valid:
        class_mrrs[superclass] = np.mean([word_mrrs[w] for w in valid])

excellent_classes_rank = sum(1 for mrr in class_mrrs.values() if mrr >= 0.8)

# -----------------------------------------------------------------------------
# METRIC 3 & 4 & 5: PERFECT WORDS, CONTAMINATION, SEMANTIC VALID
# -----------------------------------------------------------------------------

invalid_contamination = []
valid_semantic_neighbors = []

for word in ALL_CIFAR100_WORDS:
    if word not in vocab:
        continue
    true_class = word_to_class[word]
    neighbors = find_similar_words(word, vocab, embeddings, top_k=10)

    same_class_count = 0
    invalid_neighbors = []
    semantic_neighbors = []

    for neighbor_word, sim in neighbors[:5]:
        if neighbor_word in word_to_class:
            neighbor_class = word_to_class[neighbor_word]
            if neighbor_class == true_class:
                same_class_count += 1
            else:
                if word in SEMANTIC_ALLOWANCES and neighbor_class in SEMANTIC_ALLOWANCES[word]['allowed_classes']:
                    semantic_neighbors.append((neighbor_word, neighbor_class, sim))
                else:
                    invalid_neighbors.append((neighbor_word, neighbor_class, sim))

    if semantic_neighbors:
        valid_semantic_neighbors.append(word)
    if same_class_count < 3 and invalid_neighbors:
        invalid_contamination.append(word)

perfect_words = 100 - len(valid_semantic_neighbors) - len(invalid_contamination)
contamination_rate = len(invalid_contamination) / 100 * 100
semantic_valid_rate = len(valid_semantic_neighbors) / 100 * 100

# -----------------------------------------------------------------------------
# METRIC 6 & 7: MEAN SIMILARITY + CV (full vocab)
# -----------------------------------------------------------------------------

np.random.seed(42)  # Fixed seed for reproducibility
sample_size = min(100, len(vocab))
sample_indices = np.random.choice(len(vocab), size=sample_size, replace=False)
sample_vecs = embeddings[sample_indices]

norms = np.linalg.norm(sample_vecs, axis=1, keepdims=True) + 1e-12
sample_vecs_norm = sample_vecs / norms

sim_matrix = sample_vecs_norm @ sample_vecs_norm.T
upper_tri = sim_matrix[np.triu_indices(sample_size, k=1)]
mean_sim = float(np.mean(upper_tri))
std_sim = float(np.std(upper_tri))
cv = (std_sim / mean_sim) * 100

# -----------------------------------------------------------------------------
# METRIC 8: TOP-5 AVG NEIGHBOR SIMILARITY (CIFAR-100 words only)
# -----------------------------------------------------------------------------

top5_avgs = []
for word in ALL_CIFAR100_WORDS:
    if word not in vocab:
        continue
    neighbors = find_similar_words(word, vocab, embeddings, top_k=5)
    if neighbors:
        top5_avgs.append(np.mean([s for _, s in neighbors]))

top5_avg_sim = float(np.mean(top5_avgs))

# -----------------------------------------------------------------------------
# REPORT
# -----------------------------------------------------------------------------

print("=" * 60)
print("REPORT METRICS (question_5.tex -- EXP6 Final Model)")
print("=" * 60)
print(f"\n1. Overall MRR                : {overall_mrr*100:.1f}%   (report: 86.9%)")
print(f"2. Excellent superclasses     : {excellent_classes_rank}/20    (report: 16/20,  MRR>=0.8 per class)")
print(f"3. Perfect words              : {perfect_words}/100  (report: 93/100)")
print(f"4. Contamination rate         : {len(invalid_contamination)}/100  {contamination_rate:.1f}%   (report: 6/100  6.0%)")
print(f"5. Semantically valid         : {len(valid_semantic_neighbors)}/100  {semantic_valid_rate:.1f}%   (report: 1/100  1.0%)")
print(f"6. Mean similarity (sample)   : {mean_sim:.3f}  (report: 0.539)")
print(f"7. CV (std/mean * 100)        : {cv:.1f}%  (report: 23.4%)")
print(f"8. Top-5 avg neighbor sim     : {top5_avg_sim:.3f}  (report: 0.814)")

print("\n" + "-" * 60)
print("Per-class MRR (rank-based excellent = MRR >= 0.8):")
for sc, mrr in sorted(class_mrrs.items(), key=lambda x: -x[1]):
    tier = "[OK] Excellent" if mrr >= 0.8 else ("[!]  Good" if mrr >= 0.6 else "[X] Poor")
    print(f"  {tier}  {sc:40s}  MRR={mrr:.4f}")

print("=" * 60)

# -----------------------------------------------------------------------------
# t-SNE VISUALISATION (all 100 CIFAR-100 words, coloured by superclass)
# -----------------------------------------------------------------------------

def plot_tsne(vocab, embeddings, superclasses, output_file="tsne_cifar100.png"):
    """
    Generate a t-SNE plot of all 100 CIFAR-100 word embeddings,
    coloured by superclass. Saves to output_file.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
    except ImportError:
        print("\n[t-SNE] Skipped -- install matplotlib and scikit-learn to generate plot.")
        print("  pip install matplotlib scikit-learn")
        return

    # Collect vectors and labels
    words_present, vecs, class_labels = [], [], []
    for sc, members in superclasses.items():
        for word in members:
            if word in vocab:
                idx = vocab.index(word)
                vecs.append(embeddings[idx])
                words_present.append(word)
                class_labels.append(sc)

    if len(vecs) < 10:
        print("[t-SNE] Too few words found in vocabulary, skipping.")
        return

    vecs = np.array(vecs)
    # Normalise
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12)

    # Run t-SNE
    print(f"\n[t-SNE] Running on {len(vecs)} words ...")
    tsne = TSNE(n_components=2, perplexity=min(30, len(vecs) - 1),
                random_state=42, max_iter=1000, learning_rate='auto', init='pca')
    coords = tsne.fit_transform(vecs)

    # Assign a colour per superclass
    unique_classes = list(superclasses.keys())
    cmap = plt.get_cmap('tab20', len(unique_classes))
    colour_map = {sc: cmap(i) for i, sc in enumerate(unique_classes)}

    fig, ax = plt.subplots(figsize=(16, 12))
    ax.set_facecolor('#f8f8f8')

    # Plot each superclass as a separate scatter group (for legend)
    for sc in unique_classes:
        idxs = [i for i, c in enumerate(class_labels) if c == sc]
        if not idxs:
            continue
        x = coords[idxs, 0]
        y = coords[idxs, 1]
        ax.scatter(x, y, c=[colour_map[sc]], label=sc.replace('_', ' '),
                   s=120, alpha=0.85, edgecolors='white', linewidths=0.5)
        # Annotate each point with the word
        for i, idx in enumerate(idxs):
            ax.annotate(words_present[idx], (x[i], y[i]),
                        fontsize=6.5, alpha=0.8,
                        xytext=(3, 3), textcoords='offset points')

    ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
              fontsize=8, framealpha=0.9, title='Superclass')
    ax.set_title('t-SNE of CIFAR-100 Word Embeddings (128-dim Skip-Gram, 5-hop context)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('t-SNE dimension 1', fontsize=10)
    ax.set_ylabel('t-SNE dimension 2', fontsize=10)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"[t-SNE] Saved -> {output_file}")
    plt.close()


plot_tsne(vocab, embeddings, SUPERCLASSES)
