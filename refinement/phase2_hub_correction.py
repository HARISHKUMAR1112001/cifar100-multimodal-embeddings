"""
Phase 2: Advanced Refinement for the Remaining 42 Contaminated Words
=================================================================
Strategy:
1. Detect and fix "hub words" (e.g., sweetpeppers, seal, spider)
2. Apply aggressive GA to cross-domain confused words
3. Target singleton/small classes with special handling
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
del _sys, _os

import torch
import numpy as np
from tqdm import tqdm
from skipgram_trainer import SkipGramModel, find_similar_words

# ===========================================================================
# CIFAR-100 WORD STRUCTURE (Complete)
# ===========================================================================

SUPERCLASSES = {
    'aquatic_mammals': ['beaver', 'dolphin', 'otter', 'seal', 'whale'],
    'fish': ['aquariumfish', 'flatfish', 'ray', 'shark', 'trout'],
    'flowers': ['orchids', 'poppies', 'roses', 'sunflowers', 'tulips'],
    'food_containers': ['bottles', 'bowls', 'cans', 'cups', 'plates'],
    'fruits_vegetables': ['apples', 'mushrooms', 'oranges', 'pears', 'sweetpeppers'],
    'insects': ['bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach'],
    'large_carnivores': ['leopard', 'lion', 'tiger', 'wolf'],
    'large_outdoor_things': ['castle', 'skyscraper'],
    'large_natural_scenes': ['forest', 'plain', 'sea'],
    'large_omnivores_herbivores': ['camel', 'cattle', 'chimpanzee', 'kangaroo'],
    'medium_mammals': ['fox', 'porcupine', 'possum', 'raccoon', 'skunk'],
    'invertebrates': ['crab', 'lobster', 'snail', 'spider', 'worm'],
    'reptiles': ['crocodile', 'dinosaur', 'lizard', 'snake', 'turtle'],
    'small_mammals': ['hamster', 'rabbit', 'shrew', 'squirrel'],
    'trees': ['maple', 'oak', 'palm', 'pine', 'willow'],
    'vehicles_1': ['pickuptruck'],
    'vehicles_2': ['lawnmower', 'rocket', 'streetcar', 'tank', 'tractor'],
    'household_items': ['computerkeyboard', 'telephone', 'television', 'wardrobe'],
}

def get_word_to_superclass():
    """Create mapping from word to its superclass."""
    word_to_class = {}
    for superclass, members in SUPERCLASSES.items():
        for word in members:
            word_to_class[word] = superclass
    return word_to_class


WORD_TO_SUPERCLASS = get_word_to_superclass()

# Words with severe cross-domain confusion (from your test results)
# CROSS_DOMAIN_WORDS = [
#     'castle', 'dinosaur', 'flatfish', 'forest', 'hamster', 'kangaroo',
#     'lawnmower', 'lizard', 'lobster', 'maple', 'mushrooms', 'oak',
#     'otter'
# ]
# CROSS_DOMAIN_WORDS = ['ray', 'shark', 'sweetpeppers', 'computerkeyboard', 'telephone', 'television', 'bed', 'wardrobe', 'castle', 'skyscraper',
#                       'forest', 'plain', 'sea', 'camel', 'chimpanzee', 'elephant', 'kangaroo', 'crab', 'lobster', 'snail']


# Hub words needing inward pull
HUB_WORDS = ['wardrobe', 'spider', 'pickuptruck']

# Cross-domain words needing aggressive GA
CROSS_DOMAIN_WORDS = [
    'castle', 'forest', 'plain', 'sea', 'elephant',
    'computerkeyboard', 'telephone', 'television',
    'bed', 'crab', 'lobster', 'worm'
]



# ===========================================================================
# STRATEGY 1: Detect and Fix Hub Words
# ===========================================================================

def detect_hub_words(vocab, embeddings, threshold=0.85):
    """
    Find words that are abnormally similar to too many other words.
    These "hub" words contaminate neighborhoods.
    """
    print("\n" + "="*80)
    print(" DETECTING HUB WORDS (Contamination Sources)")
    print("="*80)
    
    # Normalize all embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    embeddings_norm = embeddings / norms
    
    # Compute similarity matrix
    print("Computing similarity matrix...")
    sim_matrix = embeddings_norm @ embeddings_norm.T
    
    # For each word, count how many words it's similar to (>threshold)
    hub_scores = []
    for i, word in enumerate(vocab):
        high_sim_count = np.sum(sim_matrix[i] > threshold) - 1  # Exclude self
        avg_sim = np.mean(sim_matrix[i])
        hub_scores.append((word, high_sim_count, avg_sim))
    
    # Sort by high_sim_count
    hub_scores.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 Hub Words (potential contamination sources):")
    for rank, (word, count, avg_sim) in enumerate(hub_scores[:15], 1):
        marker = "[!]" if count > 100 else "[!]" if count > 50 else ""
        print(f"  {rank:2d}. {marker} {word:20s}: similar to {count:4d} words (avg: {avg_sim:.4f})")
    
    return hub_scores

def pull_hub_words_inward(embeddings, vocab, hub_words, word_to_superclass, superclass_members, pull_strength=0.20):
    """
    Pull hub words closer to their class centroids.
    This reduces their contaminating effect on other classes.
    """
    print("\n" + "="*80)
    print(f" PULLING {len(hub_words)} HUB WORDS TOWARD THEIR CLASSES")
    print("="*80)
    
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    refined_embeddings = embeddings.copy()
    
    pulled_count = 0
    
    for hub_word in hub_words:
        if hub_word not in word_to_superclass or hub_word not in word_to_idx:
            continue
        
        true_class = word_to_superclass[hub_word]
        class_members = superclass_members[true_class]
        valid_members = [w for w in class_members if w != hub_word and w in word_to_idx]
        
        if len(valid_members) < 2:
            continue
        
        # Compute class centroid (excluding the hub word itself)
        member_indices = [word_to_idx[w] for w in valid_members]
        class_centroid = np.mean(embeddings[member_indices], axis=0)
        
        # Pull hub word toward centroid
        hub_idx = word_to_idx[hub_word]
        current_vec = refined_embeddings[hub_idx]
        
        # Weighted interpolation
        pulled_vec = (1 - pull_strength) * current_vec + pull_strength * class_centroid
        
        # Preserve norm
        pulled_vec = pulled_vec / (np.linalg.norm(pulled_vec) + 1e-10) * np.linalg.norm(current_vec)
        
        refined_embeddings[hub_idx] = pulled_vec
        pulled_count += 1
        
        print(f"  [OK] Pulled '{hub_word}' ({true_class}) toward: {', '.join(valid_members[:3])}")
    
    print(f"\n[OK] Successfully pulled {pulled_count}/{len(hub_words)} hub words")
    return refined_embeddings

# ===========================================================================
# STRATEGY 2: Aggressive GA for Cross-Domain Confusions
# ===========================================================================

def aggressive_ga_for_cross_domain(
    embeddings, vocab, word_to_superclass, superclass_members,
    cross_domain_words, generations=800, population_size=150, mutation_factor=0.10
):
    """
    Use aggressive GA settings for words with severe cross-domain confusion.
    Higher mutation, longer evolution, pure class-pull objective.
    """
    print("\n" + "="*80)
    print(f" AGGRESSIVE GA FOR {len(cross_domain_words)} CROSS-DOMAIN WORDS")
    print("="*80)
    print(f"Config: gen={generations}, pop={population_size}, mut={mutation_factor}")
    print("Objective: 95% class-pull + 5% class-push (pure clustering)")
    print("="*80 + "\n")
    
    word_to_idx = {w: i for i, w in enumerate(vocab)}
    refined_embeddings = embeddings.copy()
    
    # Pre-normalize for speed
    norms = np.linalg.norm(refined_embeddings, axis=1, keepdims=True) + 1e-10
    embeddings_norm = refined_embeddings / norms
    
    results = []
    
    for word in tqdm(cross_domain_words, desc="Aggressive GA"):
        if word not in word_to_idx or word not in word_to_superclass:
            continue
        
        word_idx = word_to_idx[word]
        true_class = word_to_superclass[word]
        same_class = [w for w in superclass_members[true_class] if w != word and w in word_to_idx]
        
        if not same_class:
            continue
        
        # Initialize population
        current_vec = refined_embeddings[word_idx]
        population = []
        
        # 70% from current position, 30% from class centroid
        same_indices = [word_to_idx[w] for w in same_class]
        class_centroid = np.mean(refined_embeddings[same_indices], axis=0)
        
        for i in range(population_size):
            if i < int(population_size * 0.7):
                # From current position with noise
                init = current_vec + np.random.randn(len(current_vec)) * 0.03
            else:
                # From class centroid with noise
                init = class_centroid + np.random.randn(len(class_centroid)) * 0.05
            population.append(init)
        
        # Evaluate initial population
        fitnesses = []
        for cand in population:
            cand_norm = cand / (np.linalg.norm(cand) + 1e-10)
            same_vecs = embeddings_norm[same_indices]
            class_sim = np.mean(same_vecs @ cand_norm)
            
            # Sample other-class words
            other_class_words = [w for w in word_to_superclass.keys()
                               if word_to_superclass[w] != true_class and w in word_to_idx]
            if len(other_class_words) > 10:
                other_sample = np.random.choice(other_class_words, 10, replace=False)
            else:
                other_sample = other_class_words
            
            other_indices = [word_to_idx[w] for w in other_sample]
            other_vecs = embeddings_norm[other_indices]
            other_sim = np.mean(other_vecs @ cand_norm)
            
            fitness = 0.95 * class_sim - 0.05 * other_sim
            fitnesses.append(fitness)
        
        best_idx = np.argmax(fitnesses)
        best_vec = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        
        initial_fitness = best_fitness
        
        # GA evolution loop
        for gen in range(generations):
            # Adaptive mutation (decays over time)
            current_mutation = mutation_factor * (1.0 - 0.3 * gen / generations)
            
            # Generate offspring
            offspring = []
            for parent in population:
                child = parent + np.random.randn(len(parent)) * current_mutation
                offspring.append(child)
            
            # Evaluate all candidates
            candidates = [best_vec] + offspring
            fitnesses = []
            
            for cand in candidates:
                cand_norm = cand / (np.linalg.norm(cand) + 1e-10)
                
                same_vecs = embeddings_norm[same_indices]
                class_sim = np.mean(same_vecs @ cand_norm)
                
                # Resample other-class words each generation for robustness
                if len(other_class_words) > 10:
                    other_sample = np.random.choice(other_class_words, 10, replace=False)
                else:
                    other_sample = other_class_words
                
                other_indices = [word_to_idx[w] for w in other_sample]
                other_vecs = embeddings_norm[other_indices]
                other_sim = np.mean(other_vecs @ cand_norm)
                
                fitness = 0.95 * class_sim - 0.05 * other_sim
                fitnesses.append(fitness)
            
            # Select best
            best_idx = np.argmax(fitnesses)
            new_best = candidates[best_idx].copy()
            new_fitness = fitnesses[best_idx]
            
            # Update best if improved
            if new_fitness > best_fitness:
                best_vec = new_best
                best_fitness = new_fitness
            
            # Elitist selection: keep top 50%
            sorted_indices = np.argsort(fitnesses)[::-1]
            elite_size = population_size // 2
            population = [candidates[i].copy() for i in sorted_indices[:elite_size]]
            
            # Repopulate with mutations of elites
            while len(population) < population_size:
                parent = population[np.random.randint(0, elite_size)]
                child = parent + np.random.randn(len(parent)) * current_mutation
                population.append(child)
        
        # Validate improvement
        initial_vec = refined_embeddings[word_idx]
        initial_norm = initial_vec / (np.linalg.norm(initial_vec) + 1e-10)
        
        same_vecs = embeddings_norm[same_indices]
        initial_class_sim = np.mean(same_vecs @ initial_norm)
        final_class_sim = np.mean(same_vecs @ (best_vec / (np.linalg.norm(best_vec) + 1e-10)))
        
        if final_class_sim > initial_class_sim:
            refined_embeddings[word_idx] = best_vec
            # Update normalized version
            embeddings_norm[word_idx] = best_vec / (np.linalg.norm(best_vec) + 1e-10)
            
            improvement = best_fitness - initial_fitness
            status = "[OK] ACCEPTED"
            results.append((word, improvement, True))
        else:
            improvement = 0.0
            status = "[!]  REJECTED"
            results.append((word, improvement, False))
        
        print(f"  {status} '{word}': {initial_fitness:.4f} -> {best_fitness:.4f} (Delta={improvement:+.4f})")
    
    # Summary
    accepted = sum(1 for _, _, acc in results if acc)
    print(f"\n{'='*80}")
    print(f"Aggressive GA Summary: {accepted}/{len(cross_domain_words)} accepted")
    print("="*80)
    
    return refined_embeddings

# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def run_phase2_refinement(input_model_path, output_model_path):
    """
    Main Phase 2 pipeline:
    1. Load Phase 1 model (fixed_hybrid_final.pth)
    2. Detect hub words
    3. Pull hub words toward their classes
    4. Apply aggressive GA to cross-domain words
    5. Save refined model
    """
    print("\n" + "="*80)
    print(" PHASE 2: ADVANCED REFINEMENT")
    print("="*80)
    print(f"Input:  {input_model_path}")
    print(f"Output: {output_model_path}")
    print(f"Current contamination: 42/76 words")
    print(f"Target contamination:  <30/76 words")
    print("="*80)
    
    # =======================================================================
    # Step 1: Load Phase 1 model
    # =======================================================================
    
    print("\n Loading Phase 1 model...")
    checkpoint = torch.load(input_model_path, map_location='cpu', weights_only=False)
    
    vocab = checkpoint['nodes']
    
    # Handle both tensor and numpy embeddings
    # if isinstance(checkpoint['embeddings'], torch.Tensor):
    #     embeddings = checkpoint['embeddings'].numpy()
    # else:
    #     embeddings = checkpoint['embeddings']
    model = SkipGramModel(checkpoint['vocab_size'], checkpoint['embedding_dim'])
    model.load_state_dict(checkpoint['model_state_dict'])
    embeddings = model.get_embeddings()
    loaded_embeddings = embeddings

    
    
    print(f"[OK] Loaded: {len(vocab)} words, {embeddings.shape[1]}-dim embeddings")
    
    # =======================================================================
    # Step 2: Detect hub words
    # =======================================================================
    
    hub_scores = detect_hub_words(vocab, embeddings, threshold=0.85)
    
    # Extract top hub words that are CIFAR-100 words
    cifar_hubs = []
    for word, count, avg_sim in hub_scores:
        if word in WORD_TO_SUPERCLASS and count > 50:
            cifar_hubs.append(word)
        if len(cifar_hubs) >= 10:  # Take top 10 CIFAR hubs
            break
    
    print(f"\n Identified {len(cifar_hubs)} CIFAR-100 hub words for correction:")
    print(f"   {', '.join(cifar_hubs)}")
    
    # =======================================================================
    # Step 3: Pull hub words toward their classes
    # =======================================================================
    
    embeddings = pull_hub_words_inward(
        embeddings, vocab, cifar_hubs, 
        WORD_TO_SUPERCLASS, SUPERCLASSES,
        pull_strength=0.25  # 25% pull toward class centroid
    )
    
    # =======================================================================
    # Step 4: Aggressive GA for cross-domain words
    # =======================================================================
    
    embeddings = aggressive_ga_for_cross_domain(
        embeddings, vocab, WORD_TO_SUPERCLASS, SUPERCLASSES,
        CROSS_DOMAIN_WORDS,
        generations=800,
        population_size=150,
        mutation_factor=0.10
    )
    
    # =======================================================================
    # Step 5: Save refined model
    # =======================================================================
    
    print("\n" + "="*80)
    print(" SAVING PHASE 2 MODEL")
    print("="*80)
    
    new_checkpoint = {
        'vocab_size': checkpoint['vocab_size'],
        'embedding_dim': checkpoint['embedding_dim'],
        'model_state_dict': checkpoint['model_state_dict'],
        'nodes': vocab,
        'embeddings': embeddings,
        'metadata': {
            'training_method': 'phase2_hub_fix_plus_aggressive_ga',
            'base_model': input_model_path,
            'phase1_contamination': 42,
            'hub_words_corrected': len(cifar_hubs),
            'cross_domain_refined': len(CROSS_DOMAIN_WORDS),
            'hub_pull_strength': 0.25,
            'aggressive_ga_config': {
                'generations': 800,
                'population': 150,
                'mutation': 0.10
            }
        }
    }
    
    torch.save(new_checkpoint, output_model_path)
    print(f"[OK] Saved to: {output_model_path}")
    
    # =======================================================================
    # Final Summary
    # =======================================================================
    
    print("\n" + "="*80)
    print(" PHASE 2 COMPLETE")
    print("="*80)
    print("Refinements applied:")
    print(f"  1. Hub word correction:     {len(cifar_hubs)} words")
    print(f"  2. Aggressive GA evolution: {len(CROSS_DOMAIN_WORDS)} words")
    print(f"\nExpected results:")
    print(f"  Contamination: 42 -> ~25-30 (target: <30)")
    print(f"  Clustering:    14 excellent -> 15-16 excellent")
    print("\n[OK] Run test1.py with '{output_model_path}' to validate!")
    print("="*80)

# ===========================================================================
# MAIN
# ===========================================================================

if __name__ == "__main__":
    INPUT_MODEL = 'fixed_hybrid_final_8_V.pth'
    OUTPUT_MODEL = 'phase2_FOR_8_VERSION_refined_final.pth'
    
    run_phase2_refinement(INPUT_MODEL, OUTPUT_MODEL)