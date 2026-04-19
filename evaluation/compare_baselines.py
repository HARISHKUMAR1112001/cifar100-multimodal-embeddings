"""
compare_baselines.py
====================
Compares our 5-hop Skip-Gram model against three standard baselines on the
CIFAR-100 semantic clustering task.

Baselines:
  1. GloVe 100d   Wikipedia + Gigaword, 6B tokens  (small, standard)
  2. GloVe 300d   Wikipedia + Gigaword, 6B tokens  (larger capacity)
  3. fastText 300d  Common Crawl, 600B tokens       (strongest general-purpose;
                     handles subword / compound terms natively)

All models are evaluated with identical metrics:
  MRR, Precision@K, Recall@K, Hit Rate@K, contamination-free count.

Downloads happen automatically via gensim on first run (one-time):
  glove-wiki-gigaword-100  ~128 MB
  glove-wiki-gigaword-300  ~376 MB
  fasttext-wiki-news-subwords-300  ~960 MB

Usage:
  python compare_baselines.py                # all four models
  python compare_baselines.py --no-fasttext  # skip fastText (saves download)
  python compare_baselines.py --no-300       # skip both 300d models
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
del _sys, _os

import argparse
import sys
import numpy as np
import gensim.downloader as api
from collections import defaultdict

#  local imports 
from skipgram_embeddings import build_my_embeddings
from skipgram_trainer import find_similar_words

# ==============================================================================
# CIFAR-100 TAXONOMY
# ==============================================================================

SUPERCLASSES = {
    'aquatic_mammals':               ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish':                          ['aquarium_fish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers':                       ['orchid', 'poppy', 'rose', 'sunflower', 'tulip'],
    'food_containers':               ['bottle', 'bowl', 'can', 'cup', 'plate'],
    'fruits_vegetables':             ['apple', 'mushroom', 'orange', 'pear', 'sweet_pepper'],
    'household_electrical_devices':  ['clock', 'keyboard', 'lamp', 'telephone', 'television'],
    'household_furniture':           ['bed', 'chair', 'couch', 'table', 'wardrobe'],
    'insects':                       ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores':              ['bear', 'leopard', 'lion', 'tiger', 'wolf'],
    'large_man_made_outdoor_things': ['bridge', 'castle', 'house', 'road', 'skyscraper'],
    'large_natural_outdoor_scenes':  ['cloud', 'forest', 'mountain', 'plain', 'sea'],
    'large_omnivores_herbivores':    ['camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo'],
    'medium_mammals':                ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'non_insect_invertebrates':      ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'people':                        ['baby', 'boy', 'girl', 'man', 'woman'],
    'reptiles':                      ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals':                 ['hamster', 'mouse', 'rabbit', 'shrew', 'squirrel'],
    'trees':                         ['maple_tree', 'oak_tree', 'palm_tree', 'pine_tree', 'willow_tree'],
    'vehicles_1':                    ['bicycle', 'bus', 'motorcycle', 'pickup_truck', 'train'],
    'vehicles_2':                    ['lawn_mower', 'rocket', 'streetcar', 'tank', 'tractor'],
}

ALL_CIFAR100_WORDS = [w for members in SUPERCLASSES.values() for w in members]
WORD_TO_CLASS = {w: sc for sc, members in SUPERCLASSES.items() for w in members}

# ==============================================================================
# EVALUATION FUNCTIONS
# ==============================================================================

def _mrr(neighbors, same_class):
    for rank, (w, _) in enumerate(neighbors, 1):
        if w in same_class:
            return 1.0 / rank
    return 0.0

def _p_at_k(neighbors, same_class, k):
    top = [w for w, _ in neighbors[:k]]
    return sum(1 for w in top if w in same_class) / k if k else 0.0

def _r_at_k(neighbors, same_class, k):
    if not same_class:
        return 0.0
    top = [w for w, _ in neighbors[:k]]
    return sum(1 for w in top if w in same_class) / len(same_class)

def _hr_at_k(neighbors, same_class, k):
    top = [w for w, _ in neighbors[:k]]
    return 1.0 if any(w in same_class for w in top) else 0.0


def evaluate(find_neighbors_fn, model_name, k_values=(1, 3, 5, 10), top_k=15):
    """
    Evaluate a model over all 100 CIFAR-100 words.

    find_neighbors_fn(word, top_k) -> [(word, score), ...]

    Returns a results dict.
    """
    mrr_scores = []
    precision  = {k: [] for k in k_values}
    recall     = {k: [] for k in k_values}
    hit_rate   = {k: [] for k in k_values}
    per_class  = defaultdict(lambda: {'mrr': [], 'p5': [], 'hr5': []})
    rank1_correct = 0
    missing = []

    for word in ALL_CIFAR100_WORDS:
        sc       = WORD_TO_CLASS[word]
        same     = set(SUPERCLASSES[sc]) - {word}
        nbrs     = find_neighbors_fn(word, top_k)

        if not nbrs:
            missing.append(word)
            continue

        mrr = _mrr(nbrs, same)
        mrr_scores.append(mrr)
        if mrr == 1.0:
            rank1_correct += 1
        per_class[sc]['mrr'].append(mrr)

        for k in k_values:
            p = _p_at_k(nbrs, same, k)
            r = _r_at_k(nbrs, same, k)
            h = _hr_at_k(nbrs, same, k)
            precision[k].append(p)
            recall[k].append(r)
            hit_rate[k].append(h)
            if k == 5:
                per_class[sc]['p5'].append(p)
                per_class[sc]['hr5'].append(h)

    mean_mrr  = float(np.mean(mrr_scores)) if mrr_scores else 0.0
    coverage  = len(mrr_scores)
    excellent = sum(
        1 for sc, v in per_class.items()
        if np.mean(v['mrr']) >= 0.8
    )

    class_rows = sorted([
        {
            'superclass': sc,
            'mrr': float(np.mean(v['mrr'])),
            'p5':  float(np.mean(v['p5']))  if v['p5']  else 0.0,
            'hr5': float(np.mean(v['hr5'])) if v['hr5'] else 0.0,
        }
        for sc, v in per_class.items()
    ], key=lambda x: x['mrr'], reverse=True)

    return {
        'model_name':    model_name,
        'mrr':           mean_mrr,
        'rank1_correct': rank1_correct,
        'coverage':      coverage,
        'precision':     {k: float(np.mean(v)) if v else 0.0 for k, v in precision.items()},
        'recall':        {k: float(np.mean(v)) if v else 0.0 for k, v in recall.items()},
        'hit_rate':      {k: float(np.mean(v)) if v else 0.0 for k, v in hit_rate.items()},
        'per_class':     class_rows,
        'excellent':     excellent,
        'missing':       missing,
    }

# ==============================================================================
# GENSIM BASELINE LOADER
# ==============================================================================

def _resolve(word, model):
    """
    Look up a word, handling multi-word terms (underscore-separated) by
    averaging component vectors. Returns normalised numpy array or None.
    """
    if word in model:
        v = model[word]
    else:
        parts = word.split('_')
        vecs = [model[p] for p in parts if p in model]
        if not vecs:
            return None
        v = np.mean(vecs, axis=0)
    norm = np.linalg.norm(v)
    return v / (norm + 1e-8)


def make_gensim_find_neighbors(model, label):
    """Build a find_neighbors function over the closed CIFAR-100 word set."""
    cifar_vecs = {}
    for w in ALL_CIFAR100_WORDS:
        v = _resolve(w, model)
        if v is not None:
            cifar_vecs[w] = v

    n_covered = len(cifar_vecs)
    n_missing = 100 - n_covered
    print(f"  [{label}] Coverage: {n_covered}/100"
          + (f"  Missing: {', '.join(w for w in ALL_CIFAR100_WORDS if w not in cifar_vecs)}" if n_missing else ""))

    def find_neighbors(word, top_k=15):
        if word not in cifar_vecs:
            return []
        q = cifar_vecs[word]
        sims = [(w, float(np.dot(q, v))) for w, v in cifar_vecs.items() if w != word]
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:top_k]

    return find_neighbors

# ==============================================================================
# OUR MODEL
# ==============================================================================

def make_our_find_neighbors():
    vocab_dict, embs = build_my_embeddings()
    vocab_list = sorted(vocab_dict.keys(), key=lambda w: vocab_dict[w])
    def find_neighbors(word, top_k=15):
        return find_similar_words(word, vocab_list, embs, top_k=top_k)
    print(f"  [Our model] vocab={len(vocab_list)}, dim={embs.shape[1]}")
    return find_neighbors

# ==============================================================================
# PRINT HELPERS
# ==============================================================================

def print_result(r, k_values=(1, 3, 5, 10)):
    print(f"\n{'='*70}")
    print(f"  {r['model_name']}")
    print(f"{'='*70}")
    print(f"  Coverage          : {r['coverage']}/100"
          + (f"  [missing: {', '.join(r['missing'])}]" if r['missing'] else ""))
    print(f"  MRR               : {r['mrr']*100:.1f}%")
    print(f"  Rank-1 correct    : {r['rank1_correct']}/100  "
          f"(nearest neighbour is same-class)")
    print(f"  Excellent SC      : {r['excellent']}/20  (MRR >= 0.8)")
    print(f"\n  {'K':>4}  {'P@K':>8}  {'R@K':>8}  {'HR@K':>8}")
    print(f"  {'-'*38}")
    for k in k_values:
        print(f"  {k:>4}  {r['precision'][k]*100:>7.1f}%"
              f"  {r['recall'][k]*100:>7.1f}%"
              f"  {r['hit_rate'][k]*100:>7.1f}%")


def print_comparison_table(results, k_values=(1, 3, 5, 10)):
    print(f"\n{'='*90}")
    print("  HEAD-TO-HEAD COMPARISON")
    print(f"{'='*90}")

    # Header
    col = 28
    hdr  = f"  {'Model':<{col}}  {'Cov':>5}  {'MRR':>7}  {'Rank-1':>7}"
    hdr += "".join(f"  {'P@'+str(k):>6}" for k in k_values)
    hdr += f"  {'HR@5':>6}  {'ExcSC':>6}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for r in results:
        name = r['model_name'][:col]
        row  = f"  {name:<{col}}  {r['coverage']:>5}  {r['mrr']*100:>6.1f}%  {r['rank1_correct']:>6}/100"
        row += "".join(f"  {r['precision'][k]*100:>5.1f}%" for k in k_values)
        row += f"  {r['hit_rate'][5]*100:>5.1f}%  {r['excellent']:>4}/20"
        print(row)

    print(f"\n  Note: 'Rank-1' = words where nearest neighbour is same-class (= MRR 1.0)")
    print(f"  Note: 'ExcSC'  = superclasses with mean MRR >= 0.8")


def print_per_class_table(results):
    print(f"\n{'='*90}")
    print("  PER-SUPERCLASS MRR")
    print(f"{'='*90}")
    hdr = f"  {'Superclass':<38}"
    for r in results:
        short = r['model_name'].split()[0]
        hdr += f"  {short:>10}"
    hdr += "  Winner"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    # Build per-class dict keyed by superclass
    class_mrr = {r['model_name']: {row['superclass']: row['mrr'] for row in r['per_class']} for r in results}

    for sc in SUPERCLASSES.keys():
        row_vals = [class_mrr[r['model_name']].get(sc, 0.0) for r in results]
        best_idx = int(np.argmax(row_vals))
        winner   = results[best_idx]['model_name'].split()[0]
        row = f"  {sc:<38}"
        for v in row_vals:
            row += f"  {v:>10.3f}"
        row += f"  {winner}"
        print(row)


def save_bar_chart(results, k_values=(1, 3, 5, 10)):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        n = len(results)
        x = np.arange(n)
        labels = [r['model_name'].split()[0] + '\n' + ' '.join(r['model_name'].split()[1:3]) for r in results]

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Left: MRR comparison
        ax = axes[0]
        bars = ax.bar(x, [r['mrr']*100 for r in results],
                      color=['#4472C4', '#ED7D31', '#A9D18E', '#FF0000'][:n], alpha=0.85)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., h + 0.4, f'{h:.1f}%',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('MRR (%)'); ax.set_title('Mean Reciprocal Rank', fontweight='bold')
        ax.set_ylim(0, 105); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.grid(axis='y', alpha=0.3)

        # Right: P@1 / P@5 / HR@5
        width = 0.25
        offsets = np.linspace(-width, width, 3)
        colors  = ['steelblue', 'coral', 'seagreen']
        metrics = [('P@1', [r['precision'][1]*100 for r in results]),
                   ('P@5', [r['precision'][5]*100 for r in results]),
                   ('HR@5',[r['hit_rate'][5]*100  for r in results])]
        ax = axes[1]
        for (label, vals), off, col in zip(metrics, offsets, colors):
            bars = ax.bar(x + off, vals, width, label=label, color=col, alpha=0.85)
            for bar in bars:
                h = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., h + 0.4, f'{h:.0f}',
                        ha='center', va='bottom', fontsize=7)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Score (%)'); ax.set_title('Precision@K / Hit Rate@5', fontweight='bold')
        ax.set_ylim(0, 110); ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0f}%'))
        ax.legend(fontsize=9); ax.grid(axis='y', alpha=0.3)

        fig.suptitle('CIFAR-100 Semantic Clustering  Baseline Comparison', fontsize=13, fontweight='bold')
        fig.tight_layout()
        out = 'baseline_comparison.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"\n  Chart saved  {out}")
    except ImportError:
        print("  (matplotlib not available  skipping chart)")


def save_heatmap(results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        superclass_order = list(SUPERCLASSES.keys())
        class_mrr = {r['model_name']: {row['superclass']: row['mrr'] for row in r['per_class']} for r in results}
        heat_data = np.array([
            [class_mrr[r['model_name']].get(sc, 0.0) for sc in superclass_order]
            for r in results
        ])
        ylabels = [r['model_name'].split()[0] + ' ' + ' '.join(r['model_name'].split()[1:2]) for r in results]

        fig, ax = plt.subplots(figsize=(16, max(3, len(results) * 1.2)))
        im = ax.imshow(heat_data, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
        plt.colorbar(im, ax=ax, shrink=0.8, label='MRR')
        ax.set_yticks(range(len(results))); ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xticks(range(len(superclass_order)))
        ax.set_xticklabels([s.replace('_', '\n') for s in superclass_order], fontsize=7)
        for i in range(heat_data.shape[0]):
            for j in range(heat_data.shape[1]):
                v = heat_data[i, j]
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6,
                        color='black' if 0.25 < v < 0.85 else 'white')
        ax.set_title('Per-Superclass MRR Heatmap', fontsize=12, fontweight='bold')
        fig.tight_layout()
        out = 'baseline_comparison_heatmap.png'
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Heatmap saved  {out}")
    except ImportError:
        pass

# ==============================================================================
# MAIN
# ==============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare CIFAR-100 embedding baselines')
    parser.add_argument('--no-fasttext', action='store_true', help='Skip fastText (saves ~960 MB download)')
    parser.add_argument('--no-300',      action='store_true', help='Skip all 300d models')
    parser.add_argument('--no-charts',   action='store_true', help='Skip saving charts')
    args = parser.parse_args()

    print("\n" + "="*70)
    print("  LOADING MODELS")
    print("="*70)

    all_results = []

    #  GloVe 100d 
    print("\nLoading GloVe 100d (Wikipedia+Gigaword, ~128 MB on first run) ...")
    glove_100 = api.load("glove-wiki-gigaword-100")
    fn_g100   = make_gensim_find_neighbors(glove_100, "GloVe-100d")
    all_results.append(evaluate(fn_g100, "GloVe 100d"))

    #  GloVe 300d 
    if not args.no_300:
        print("\nLoading GloVe 300d (~376 MB on first run) ...")
        glove_300 = api.load("glove-wiki-gigaword-300")
        fn_g300   = make_gensim_find_neighbors(glove_300, "GloVe-300d")
        all_results.append(evaluate(fn_g300, "GloVe 300d"))

    #  fastText 300d 
    if not args.no_fasttext and not args.no_300:
        print("\nLoading fastText 300d (Common Crawl, ~960 MB on first run) ...")
        print("  [This is the strongest general-purpose baseline]")
        ft_300 = api.load("fasttext-wiki-news-subwords-300")
        fn_ft  = make_gensim_find_neighbors(ft_300, "fastText-300d")
        all_results.append(evaluate(fn_ft, "fastText 300d (CC)"))

    #  Our model 
    print("\nLoading our model (5-hop Skip-Gram + VG corpus + 5-phase GA) ...")
    fn_ours = make_our_find_neighbors()
    all_results.append(evaluate(fn_ours, "Ours (5-hop VG+GA 128d)"))

    #  Print results 
    for r in all_results:
        print_result(r)

    print_comparison_table(all_results)
    print_per_class_table(all_results)

    if not args.no_charts:
        print("\nSaving charts ...")
        save_bar_chart(all_results)
        save_heatmap(all_results)

    #  Win count summary 
    print(f"\n{'='*70}")
    print("  WIN SUMMARY (per superclass, which model has highest MRR)")
    print(f"{'='*70}")
    win_counts = defaultdict(int)
    for sc in SUPERCLASSES.keys():
        best_mrr  = -1
        best_name = ""
        for r in all_results:
            class_map = {row['superclass']: row['mrr'] for row in r['per_class']}
            v = class_map.get(sc, 0.0)
            if v > best_mrr:
                best_mrr  = v
                best_name = r['model_name'].split()[0]
        win_counts[best_name] += 1
    for name, count in sorted(win_counts.items(), key=lambda x: -x[1]):
        print(f"  {name:<30} {count:>3} / 20 superclasses")

    print(f"\n{'='*70}")
    print("  Done.")
    print(f"{'='*70}")
