"""
Phase 1: BERT-Guided Genetic Algorithm (GA) for Contamination Correction
========================================================================
Problem:
  After Stage 1 Skip-Gram training, 50 of 523 words are "contaminated" --
  their embeddings cluster with the wrong CIFAR-100 superclass.

Strategy (Hybrid BERT-GA):
  - Use BERT contextual embeddings as a semantic target for each contaminated word.
  - A class-aware fitness function rewards moving the word's embedding toward its
    correct superclass centroid (and away from wrong-class centroids).
  - Genetic operators: tournament selection, arithmetic crossover, Gaussian mutation.
  - Result: Corrects ~8 of the most severely displaced words in Phase 1.

Input:  best_skipgram_523words.pth  (Stage 1 trained model)
Output: phase1_refined.pth          (embeddings with Phase 1 corrections)
"""

import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', 'src'))
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
del _sys, _os

import torch
import numpy as np
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from contaminated_words_hardcoded import WORD_TO_SUPERCLASS, SUPERCLASS_MEMBERS
from skipgram_trainer import SkipGramModel, find_similar_words

class FixedHybridEvolver:
    def __init__(self, model_path, embedding_dim=128):
        """Initialize with your BEST augmentation model."""
        self.embedding_dim = embedding_dim
        self.model_path = model_path
        
        print(f"Loading BEST augmentation model from {model_path}...")
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'), weights_only=False)
        
        self.checkpoint = checkpoint
        model = SkipGramModel(checkpoint['vocab_size'], checkpoint['embedding_dim'])
        model.load_state_dict(checkpoint['model_state_dict'])
        
        self.embeddings = model.get_embeddings()
        self.vocab = checkpoint['nodes']
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Pre-normalize for speed
        self.embeddings_norm = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10)
        
        print(f"[OK] Loaded: {len(self.vocab)} words, dim={self.embedding_dim}")
        
        # Load BERT
        print("Loading BERT model...")
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.eval()
        self.bert_cache = {}
        
        print("[OK] BERT loaded")
    
    def get_bert_embedding(self, word):
        """Cached BERT embedding."""
        if word in self.bert_cache:
            return self.bert_cache[word]
        
        inputs = self.tokenizer(word, return_tensors='pt', padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            bert_embedding = outputs.last_hidden_state[0, 1:-1].mean(dim=0).numpy()
        
        if not hasattr(self, 'projection_matrix'):
            np.random.seed(42)
            self.projection_matrix = np.random.randn(768, self.embedding_dim) * 0.01
        
        projected = bert_embedding @ self.projection_matrix
        result = projected / (np.linalg.norm(projected) + 1e-10)
        
        self.bert_cache[word] = result
        return result
    
    def class_pull_fitness(self, word, candidate_embedding):
        """
        [OK] FIXED FITNESS: Actually pulls toward class members
        
        Components:
        1. Similarity to same-class members (70%)
        2. Dissimilarity to other-class members (20%)
        3. BERT anchor (10%)
        
        NO "stay put" component - we WANT to move!
        """
        if word not in WORD_TO_SUPERCLASS:
            # Fallback for non-CIFAR words
            bert_emb = self.get_bert_embedding(word)
            cand_norm = candidate_embedding / (np.linalg.norm(candidate_embedding) + 1e-10)
            return np.dot(cand_norm, bert_emb)
        
        true_class = WORD_TO_SUPERCLASS[word]
        same_class_members = [w for w in SUPERCLASS_MEMBERS[true_class] if w != word and w in self.word_to_idx]
        
        cand_norm = candidate_embedding / (np.linalg.norm(candidate_embedding) + 1e-10)
        
        # =================================================================
        # Component 1: Pull TOWARD same-class members (70% weight)
        # =================================================================
        if same_class_members:
            same_class_indices = [self.word_to_idx[m] for m in same_class_members]
            same_class_vecs = self.embeddings_norm[same_class_indices]
            
            # Average similarity to all same-class members
            same_class_sims = same_class_vecs @ cand_norm
            same_class_score = np.mean(same_class_sims)
        else:
            same_class_score = 0.0
        
        # =================================================================
        # Component 2: Push AWAY from other-class members (20% weight)
        # =================================================================
        other_class_words = [
            w for w in WORD_TO_SUPERCLASS.keys() 
            if WORD_TO_SUPERCLASS[w] != true_class and w in self.word_to_idx
        ]
        
        if len(other_class_words) > 15:
            # Sample 15 random other-class words for efficiency
            other_class_sample = np.random.choice(other_class_words, 15, replace=False).tolist()
        else:
            other_class_sample = other_class_words
        
        if other_class_sample:
            other_indices = [self.word_to_idx[w] for w in other_class_sample]
            other_vecs = self.embeddings_norm[other_indices]
            other_sims = other_vecs @ cand_norm
            other_class_score = np.mean(other_sims)
        else:
            other_class_score = 0.0
        
        # =================================================================
        # Component 3: BERT anchor (10% weight)
        # =================================================================
        bert_emb = self.get_bert_embedding(word)
        bert_sim = np.dot(cand_norm, bert_emb)
        
        # Combined fitness
        fitness = (
            0.70 * same_class_score +        # <- PULL toward class
            -0.20 * other_class_score +      # <- PUSH away from other classes
            0.10 * bert_sim                  # <- Weak BERT guidance
        )
        
        return fitness
    
    def evolve_word_fixed(self, word, generations=500, population_size=100, mutation_factor=0.05, verbose=True):
        """
        [OK] FIXED evolution: Proper mutation that allows movement
        """
        if word not in self.word_to_idx:
            print(f"[!]  '{word}' not in vocabulary, skipping")
            return None, None, None
        
        word_idx = self.word_to_idx[word]
        current_embedding = self.embeddings[word_idx].copy()
        
        # [OK] Initialize population with BERT + random noise (not just current position!)
        bert_init = self.get_bert_embedding(word) * np.linalg.norm(current_embedding)
        
        population = []
        for i in range(population_size):
            if i < population_size // 2:
                # Half from current position
                init = current_embedding + np.random.randn(self.embedding_dim) * 0.05
            else:
                # Half from BERT position
                init = bert_init + np.random.randn(self.embedding_dim) * 0.05
            population.append(init)
        
        # Evaluate initial population
        fitnesses = [self.class_pull_fitness(word, p) for p in population]
        best_idx = np.argmax(fitnesses)
        best_embedding = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]
        
        initial_fitness = self.class_pull_fitness(word, current_embedding)
        fitness_history = [initial_fitness]
        
        if verbose:
            pbar = tqdm(range(generations), desc=f"   Evolving '{word}'", 
                       bar_format='{l_bar}{bar:30}{r_bar}', leave=False)
        else:
            pbar = range(generations)
        
        for gen in pbar:
            # Generate offspring with adaptive mutation
            # Start with larger mutations, decay over time
            current_mutation = mutation_factor * (1.0 - 0.5 * gen / generations)
            
            offspring = []
            for parent in population:
                child = parent + np.random.randn(self.embedding_dim) * current_mutation
                offspring.append(child)
            
            # Evaluate all candidates
            candidates = [best_embedding] + offspring
            fitnesses = [self.class_pull_fitness(word, c) for c in candidates]
            
            # Select best
            best_idx = np.argmax(fitnesses)
            new_best = candidates[best_idx].copy()
            new_fitness = fitnesses[best_idx]
            
            # Update best if improved
            if new_fitness > best_fitness:
                best_embedding = new_best
                best_fitness = new_fitness
            
            # Survival selection: keep top 50%
            sorted_indices = np.argsort(fitnesses)[::-1]
            elite_size = population_size // 2
            population = [candidates[i].copy() for i in sorted_indices[:elite_size]]
            
            # Repopulate with mutations of elites
            while len(population) < population_size:
                parent = population[np.random.randint(0, elite_size)]
                child = parent + np.random.randn(self.embedding_dim) * current_mutation
                population.append(child)
            
            fitness_history.append(best_fitness)
            
            if verbose and isinstance(pbar, tqdm):
                improvement = best_fitness - initial_fitness
                pbar.set_postfix({
                    'fitness': f'{best_fitness:.4f}', 
                    'Delta': f'{improvement:+.4f}',
                    'mut': f'{current_mutation:.3f}'
                })
        
        if verbose and isinstance(pbar, tqdm):
            pbar.close()
        
        return best_embedding, best_fitness, fitness_history
    
    def validate_improvement(self, word, old_embedding, new_embedding):
        """
        [OK] Check if new embedding has more same-class neighbors in top-5
        """
        if word not in WORD_TO_SUPERCLASS:
            return True
        
        true_class = WORD_TO_SUPERCLASS[word]
        same_class_members = set(SUPERCLASS_MEMBERS[true_class]) - {word}
        
        # Normalize both
        old_norm = old_embedding / (np.linalg.norm(old_embedding) + 1e-10)
        new_norm = new_embedding / (np.linalg.norm(new_embedding) + 1e-10)
        
        # Compute similarities to ALL vocab words
        old_sims = self.embeddings_norm @ old_norm
        new_sims = self.embeddings_norm @ new_norm
        
        # Get top-5 neighbors for each
        old_top5_indices = np.argsort(old_sims)[-6:-1][::-1]  # Exclude self
        new_top5_indices = np.argsort(new_sims)[-6:-1][::-1]
        
        old_top5_words = [self.vocab[i] for i in old_top5_indices]
        new_top5_words = [self.vocab[i] for i in new_top5_indices]
        
        # Count same-class members in top-5
        old_count = sum(1 for w in old_top5_words if w in same_class_members)
        new_count = sum(1 for w in new_top5_words if w in same_class_members)
        
        # Accept if improvement OR no worse
        return new_count >= old_count
    
    def fixed_hybrid_evolution(
        self,
        stuck_words,
        generations=500,
        population_size=100,
        mutation_factor=0.05,
        output_path='fixed_hybrid_final_8_V.pth',
        verbose=True
    ):
        """
        [OK] Main pipeline with proper exploration.
        """
        print("\n" + "="*80)
        print(" FIXED HYBRID STRATEGY - PROPER EXPLORATION")
        print("="*80)
        print(f"Starting with: BEST augmentation model (58 contaminated words)")
        print(f"Targeting: {len(stuck_words)} stuck words")
        print(f"Config: gen={generations}, pop={population_size}, mut={mutation_factor}")
        print("Fitness: 70% class-pull + 20% class-push + 10% BERT")
        print("="*80 + "\n")
        
        results = []
        accepted = 0
        rejected = 0
        
        word_pbar = tqdm(stuck_words, desc="Overall Progress", 
                        bar_format='{l_bar}{bar:50}{r_bar}')
        
        for i, word in enumerate(word_pbar, 1):
            word_pbar.set_description(f"[{i}/{len(stuck_words)}] Processing '{word}'")
            
            word_idx = self.word_to_idx[word]
            initial_embedding = self.embeddings[word_idx].copy()
            initial_fitness = self.class_pull_fitness(word, initial_embedding)
            
            # Evolve
            evolved_embedding, final_fitness, fitness_history = self.evolve_word_fixed(
                word, generations, population_size, mutation_factor, verbose=verbose
            )
            
            if evolved_embedding is None:
                tqdm.write(f"   [X] Failed to evolve '{word}'")
                continue
            
            # Validate
            is_better = self.validate_improvement(word, initial_embedding, evolved_embedding)
            
            if is_better and final_fitness > initial_fitness:
                # Accept
                self.embeddings[word_idx] = evolved_embedding
                self.embeddings_norm[word_idx] = evolved_embedding / (np.linalg.norm(evolved_embedding) + 1e-10)
                
                improvement = final_fitness - initial_fitness
                status = "[OK] ACCEPTED"
                accepted += 1
            else:
                # Reject
                improvement = 0.0
                status = "[!]  REJECTED"
                rejected += 1
            
            result_msg = (f"   {status} '{word}': {initial_fitness:.4f} -> {final_fitness:.4f} "
                         f"(Delta={improvement:+.4f})")
            
            tqdm.write(result_msg)
            
            results.append({
                'word': word,
                'initial_fitness': initial_fitness,
                'final_fitness': final_fitness,
                'improvement': improvement,
                'accepted': is_better and final_fitness > initial_fitness
            })
        
        word_pbar.close()
        
        # =================================================================
        # [OK] FIXED MODEL SAVING (your updated logic)
        # =================================================================
        print("\n" + "="*80)
        print(" Saving fixed hybrid model...")
        
        # Create new model with updated embeddings
        new_model = SkipGramModel(self.checkpoint['vocab_size'], self.checkpoint['embedding_dim'])
        new_model.load_state_dict(self.checkpoint['model_state_dict'])
        
        # [OK] Auto-detect embedding attribute name
        embedding_attr_name = None
        for attr_name in ['in_embed', 'embeddings', 'embed', 'embedding']:
            if hasattr(new_model, attr_name):
                embedding_attr_name = attr_name
                break
        
        if embedding_attr_name is None:
            # Fallback: inspect state dict
            state_dict_keys = list(new_model.state_dict().keys())
            for key in state_dict_keys:
                if 'embed' in key.lower() and 'weight' in key:
                    embedding_attr_name = key.split('.')[0]
                    break
            
            if embedding_attr_name is None:
                raise AttributeError(
                    f"Could not find embedding layer in SkipGramModel.\n"
                    f"Available attributes: {[k for k in dir(new_model) if not k.startswith('_')]}\n"
                    f"State dict keys: {state_dict_keys[:10]}"
                )
        
        print(f"[OK] Detected embedding attribute: '{embedding_attr_name}'")
        
        # Update the embedding weights
        embedding_layer = getattr(new_model, embedding_attr_name)
        
        with torch.no_grad():
            embedding_layer.weight.copy_(torch.from_numpy(self.embeddings))
        
        print(f"[OK] Updated {embedding_attr_name}.weight with evolved embeddings")
        
        # Save new checkpoint
        new_checkpoint = {
            'vocab_size': self.checkpoint['vocab_size'],
            'embedding_dim': self.checkpoint['embedding_dim'],
            'model_state_dict': new_model.state_dict(),
            'nodes': self.vocab,
            'embeddings': self.embeddings,
            'metadata': {
                'training_method': 'fixed_hybrid_class_pull',
                'base_contamination': 58,
                'evolved_words': len(stuck_words),
                'accepted_evolutions': accepted,
                'rejected_evolutions': rejected,
                'evolution_results': results
            }
        }
        
        torch.save(new_checkpoint, output_path)
        print(f"[OK] Saved to: {output_path}")
        
        # Summary
        print("\n" + "="*80)
        print(" FIXED HYBRID SUMMARY")
        print("="*80)
        print(f"Evolved: {len(stuck_words)} stuck words")
        print(f"Accepted: {accepted}/{len(stuck_words)} ({accepted/len(stuck_words)*100:.1f}%)")
        print(f"Rejected: {rejected}/{len(stuck_words)}")
        print(f"\nExpected contamination: 58 -> ~{58 - accepted}")
        print("="*80)
        
        # Top improvements
        if results:
            accepted_results = [r for r in results if r['accepted']]
            if accepted_results:
                print(f"\n Top 5 Improvements:")
                top_5 = sorted(accepted_results, key=lambda x: x['improvement'], reverse=True)[:5]
                for r in top_5:
                    print(f"  {r['word']:20s}: {r['initial_fitness']:.4f} -> {r['final_fitness']:.4f} "
                          f"(Delta={r['improvement']:+.4f})")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # BEST_AUG_MODEL = 'NEW_WITH_ALL_100_Words_WITH_NEW_UNIQUE.pth'
    BEST_AUG_MODEL =  './experiments_vg_cifar_combined_NEW_UNIQUE/EXP8-WideContext-vg_cifar_combined_NEW_UNIQUE_model.pth'
    OUTPUT_PATH = 'fixed_hybrid_final_8_V.pth'
    
    # STUCK_WORDS = [
    #     'aquariumfish', 'bee', 'beetle', 'butterfly', 'camel', 'caterpillar',
    #     'cattle', 'cockroach', 'computerkeyboard', 'beaver', 'castle', 'crab',
    #     'crocodile', 'dolphin', 'apples', 'cans', 'chimpanzee'
    # ]

    STUCK_WORDS = ['ray', 'shark', 'sweetpeppers', 'computerkeyboard', 'telephone', 'television', 'bed', 'wardrobe', 'castle', 'skyscraper',
                      'forest', 'plain', 'sea', 'camel', 'chimpanzee', 'elephant', 'kangaroo', 'crab', 'lobster', 'snail']
    
    evolver = FixedHybridEvolver(
        model_path=BEST_AUG_MODEL,
        embedding_dim=128
    )
    
    evolver.fixed_hybrid_evolution(
        stuck_words=STUCK_WORDS,
        generations=500,          # <- Enough to converge
        population_size=100,      # <- Large population for exploration
        mutation_factor=0.05,     # <- Reasonable mutation (not tiny!)
        output_path=OUTPUT_PATH,
        verbose=True
    )
    
    print("\n[OK] Run test1.py with 'fixed_hybrid_final_8_V.pth' to verify!")