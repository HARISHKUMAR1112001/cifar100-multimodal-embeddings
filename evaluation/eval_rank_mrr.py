"""
Rank-Based Semantic Evaluation for CIFAR-100 Embeddings
========================================================
Tests: How many same-class words appear in top-K neighbors?
Metrics: 
  - Precision@K (P@K)
  - Mean Reciprocal Rank (MRR)
  - Recall@K
  - Hit Rate@K
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
del _sys, _os

import numpy as np
import torch
from collections import defaultdict
from skipgram_embeddings import build_my_embeddings
from eval_cosine_threshold import SUPERCLASSES, ALL_CIFAR100_WORDS, find_similar_words

# ===========================================================================
# LOAD MODEL
# ===========================================================================


vocab_dict, embeddings = build_my_embeddings()  # [OK] CORRECT - 2 values
vocab = sorted(vocab_dict.keys(), key=lambda w: vocab_dict[w])  # [OK] Create list from dict

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def get_word_to_superclass():
    """Map each word to its superclass."""
    word_to_class = {}
    for superclass, members in SUPERCLASSES.items():
        for word in members:
            word_to_class[word] = superclass
    return word_to_class

def get_class_members(word, word_to_class):
    """Get all members of the same class (excluding the word itself)."""
    if word not in word_to_class:
        return []
    
    superclass = word_to_class[word]
    members = SUPERCLASSES.get(superclass, [])
    return [w for w in members if w != word]

# ===========================================================================
# RANK-BASED METRICS
# ===========================================================================

def compute_precision_at_k(neighbors, same_class_words, k):
    """
    Precision@K: What fraction of top-K neighbors are same-class?
    P@K = (# same-class in top-K) / K
    """
    if k == 0 or not neighbors:
        return 0.0
    
    top_k_words = [w for w, _ in neighbors[:k]]
    same_class_in_top_k = sum(1 for w in top_k_words if w in same_class_words)
    
    return same_class_in_top_k / k

def compute_recall_at_k(neighbors, same_class_words, k):
    """
    Recall@K: What fraction of same-class words appear in top-K?
    R@K = (# same-class in top-K) / (total # same-class words)
    """
    if not same_class_words or not neighbors:
        return 0.0
    
    top_k_words = [w for w, _ in neighbors[:k]]
    same_class_in_top_k = sum(1 for w in top_k_words if w in same_class_words)
    
    return same_class_in_top_k / len(same_class_words)

def compute_mean_reciprocal_rank(neighbors, same_class_words):
    """
    Mean Reciprocal Rank (MRR): Average of 1/rank for first same-class word.
    MRR = 1 / (rank of first same-class word)
    
    Example:
      - First same-class at rank 1 -> MRR = 1.0
      - First same-class at rank 2 -> MRR = 0.5
      - First same-class at rank 5 -> MRR = 0.2
      - No same-class found -> MRR = 0.0
    """
    if not neighbors or not same_class_words:
        return 0.0
    
    for rank, (word, _) in enumerate(neighbors, start=1):
        if word in same_class_words:
            return 1.0 / rank
    
    return 0.0  # No same-class word found

def compute_hit_rate_at_k(neighbors, same_class_words, k):
    """
    Hit Rate@K: Does at least one same-class word appear in top-K?
    HR@K = 1 if any same-class in top-K, else 0
    """
    if not neighbors or not same_class_words:
        return 0.0
    
    top_k_words = [w for w, _ in neighbors[:k]]
    return 1.0 if any(w in same_class_words for w in top_k_words) else 0.0

# ===========================================================================
# RANK-BASED EVALUATION
# ===========================================================================

def evaluate_rank_based_metrics(k_values=[1, 3, 5, 10]):
    """
    Comprehensive rank-based evaluation for all CIFAR-100 words.
    """
    
    print("="*80)
    print("RANK-BASED SEMANTIC EVALUATION")
    print("="*80)
    print(f"Testing: {len(ALL_CIFAR100_WORDS)} CIFAR-100 words")
    print(f"Metrics: Precision@K, Recall@K, MRR, Hit Rate@K")
    print(f"K values: {k_values}")
    print("="*80)
    
    word_to_class = get_word_to_superclass()
    
    # Storage for metrics
    results = {
        'precision': {k: [] for k in k_values},
        'recall': {k: [] for k in k_values},
        'mrr': [],
        'hit_rate': {k: [] for k in k_values}
    }
    
    # Per-class metrics
    class_results = defaultdict(lambda: {
        'precision': {k: [] for k in k_values},
        'recall': {k: [] for k in k_values},
        'mrr': [],
        'hit_rate': {k: [] for k in k_values}
    })
    
    # Evaluate each word
    print("\n" + "="*80)
    print("INDIVIDUAL WORD EVALUATION")
    print("="*80)
    
    for i, word in enumerate(ALL_CIFAR100_WORDS, 1):
        if word not in vocab:
            print(f"[{i:3d}/100] [X] '{word}' not in vocabulary")
            continue
        
        superclass = word_to_class.get(word, "unknown")
        same_class_words = get_class_members(word, word_to_class)
        
        if not same_class_words:
            print(f"[{i:3d}/100] [!]  '{word}' has no class members")
            continue
        
        # Get top-10 neighbors
        neighbors = find_similar_words(word, vocab, embeddings, top_k=15)
        
        if not neighbors:
            print(f"[{i:3d}/100] [!]  '{word}' has no neighbors")
            continue
        
        # Compute MRR
        mrr = compute_mean_reciprocal_rank(neighbors, same_class_words)
        results['mrr'].append(mrr)
        class_results[superclass]['mrr'].append(mrr)
        
        # Compute P@K, R@K, HR@K for each K
        metrics_str = []
        for k in k_values:
            p_at_k = compute_precision_at_k(neighbors, same_class_words, k)
            r_at_k = compute_recall_at_k(neighbors, same_class_words, k)
            hr_at_k = compute_hit_rate_at_k(neighbors, same_class_words, k)
            
            results['precision'][k].append(p_at_k)
            results['recall'][k].append(r_at_k)
            results['hit_rate'][k].append(hr_at_k)
            
            class_results[superclass]['precision'][k].append(p_at_k)
            class_results[superclass]['recall'][k].append(r_at_k)
            class_results[superclass]['hit_rate'][k].append(hr_at_k)
            
            metrics_str.append(f"P@{k}={p_at_k:.2f}")
        
        # Print individual results
        first_neighbor = neighbors[0][0] if neighbors else "none"
        same_class_marker = "[OK]" if first_neighbor in same_class_words else "[X]"
        
        print(f"[{i:3d}/100] {same_class_marker} '{word:20s}' (class: {superclass:25s}) MRR={mrr:.3f}  {', '.join(metrics_str)}")
        print(f"          Top-5: {', '.join([f'{w}({s:.2f})' for w, s in neighbors[:5]])}")
    
    # =======================================================================
    # AGGREGATE RESULTS
    # =======================================================================
    
    print("\n" + "="*80)
    print("OVERALL RESULTS (ALL 100 WORDS)")
    print("="*80)
    
    print(f"\n Mean Reciprocal Rank (MRR): {np.mean(results['mrr']):.4f}")
    print(f"   -> Average rank of first same-class neighbor: {1/np.mean(results['mrr']):.2f}")
    
    print(f"\n Precision@K (fraction of top-K that are same-class):")
    for k in k_values:
        mean_p = np.mean(results['precision'][k])
        print(f"   P@{k:2d} = {mean_p:.4f} ({mean_p*100:.1f}%)")
    
    print(f"\n Recall@K (fraction of same-class found in top-K):")
    for k in k_values:
        mean_r = np.mean(results['recall'][k])
        print(f"   R@{k:2d} = {mean_r:.4f} ({mean_r*100:.1f}%)")
    
    print(f"\n Hit Rate@K (% words with >=1 same-class in top-K):")
    for k in k_values:
        mean_hr = np.mean(results['hit_rate'][k])
        print(f"   HR@{k:2d} = {mean_hr:.4f} ({mean_hr*100:.1f}%)")
    
    # =======================================================================
    # PER-CLASS RESULTS
    # =======================================================================
    
    print("\n" + "="*80)
    print("PER-CLASS RESULTS (20 SUPERCLASSES)")
    print("="*80)
    
    class_summary = []
    for superclass, metrics in class_results.items():
        mrr = np.mean(metrics['mrr']) if metrics['mrr'] else 0.0
        p_at_5 = np.mean(metrics['precision'][5]) if metrics['precision'][5] else 0.0
        hr_at_5 = np.mean(metrics['hit_rate'][5]) if metrics['hit_rate'][5] else 0.0
        
        class_summary.append((superclass, mrr, p_at_5, hr_at_5))
    
    # Sort by MRR
    class_summary.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n{'Superclass':35s} {'MRR':>8s} {'P@5':>8s} {'HR@5':>8s}")
    print("-"*80)
    for superclass, mrr, p_at_5, hr_at_5 in class_summary:
        status = "[OK]" if mrr >= 0.8 else ("[!]" if mrr >= 0.6 else "[X]")
        print(f"{status} {superclass:35s} {mrr:7.4f}  {p_at_5:7.4f}  {hr_at_5:7.4f}")
    
    # =======================================================================
    # QUALITY TIERS
    # =======================================================================
    
    print("\n" + "="*80)
    print("QUALITY TIER BREAKDOWN")
    print("="*80)
    
    mrr_values = [mrr for _, mrr, _, _ in class_summary]
    
    excellent = sum(1 for mrr in mrr_values if mrr >= 0.8)
    good = sum(1 for mrr in mrr_values if 0.6 <= mrr < 0.8)
    fair = sum(1 for mrr in mrr_values if 0.4 <= mrr < 0.6)
    poor = sum(1 for mrr in mrr_values if mrr < 0.4)
    
    print(f"[OK] Excellent (MRR >= 0.8): {excellent}/20 ({excellent/20*100:.1f}%)")
    print(f"[OK] Good (0.6 <= MRR < 0.8): {good}/20 ({good/20*100:.1f}%)")
    print(f"[!]  Fair (0.4 <= MRR < 0.6): {fair}/20 ({fair/20*100:.1f}%)")
    print(f"[X] Poor (MRR < 0.4): {poor}/20 ({poor/20*100:.1f}%)")
    
    # =======================================================================
    # FINAL SUMMARY
    # =======================================================================
    
    print("\n" + "="*80)
    print(" FINAL RANK-BASED QUALITY ASSESSMENT")
    print("="*80)
    
    overall_mrr = np.mean(results['mrr'])
    overall_p5 = np.mean(results['precision'][5])
    overall_hr5 = np.mean(results['hit_rate'][5])
    
    print(f"\n Overall MRR: {overall_mrr:.4f}")
    print(f"   -> Avg rank of 1st same-class: {1/overall_mrr:.2f}")
    
    print(f"\n Overall Precision@5: {overall_p5:.4f}")
    print(f"   -> {overall_p5*5:.1f}/5 neighbors are same-class on average")
    
    print(f"\n Overall Hit Rate@5: {overall_hr5:.4f}")
    print(f"   -> {overall_hr5*100:.1f}% of words have >=1 same-class in top-5")
    
    # Benchmarking
    print(f"\n BENCHMARK COMPARISON:")
    
    if overall_mrr >= 0.8:
        print(f"   ***** OUTSTANDING (MRR >= 0.8)")
    elif overall_mrr >= 0.6:
        print(f"   **** EXCELLENT (MRR >= 0.6)")
    elif overall_mrr >= 0.4:
        print(f"   *** GOOD (MRR >= 0.4)")
    else:
        print(f"   ** FAIR (MRR < 0.4)")
    
    if overall_hr5 >= 0.95:
        print(f"   [OK] NEAR-PERFECT Hit Rate@5 (>=95%)")
    elif overall_hr5 >= 0.85:
        print(f"   [OK] EXCELLENT Hit Rate@5 (>=85%)")
    elif overall_hr5 >= 0.70:
        print(f"   [!]  GOOD Hit Rate@5 (>=70%)")
    else:
        print(f"   [X] NEEDS IMPROVEMENT Hit Rate@5 (<70%)")
    
    print("="*80)

# ===========================================================================
# RUN EVALUATION
# ===========================================================================

if __name__ == "__main__":
    evaluate_rank_based_metrics(k_values=[1, 3, 5, 10])