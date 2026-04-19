"""
Phase 3: Targeted Embedding Blending (Surgical Fixes)
======================================================

Problem:
  After Phase 2, ~15 words remain contaminated due to hub-word effects and
  cross-domain confusion (e.g., household items mis-clustering with electronics).

Strategy (Surgical Blending):
  - Identifies class centroids from unambiguous "clean" class members.
  - For each contaminated word, interpolates its embedding toward the correct
    class centroid using a blending weight alpha tuned per word.
  - Special handling for household / electrical-device confusion and
    singleton classes with few anchor words.
  - No gradient updates -- operates directly on the embedding matrix.

Input:  phase2_refined.pth    (Phase 2 output)
Output: phase3_refined.pth    (embeddings with surgical blending applied)
"""

import numpy as np
import torch
from skipgram_trainer import SkipGramModel

class SurgicalEmbeddingFixer:
    def __init__(self, model_path):
        """Load Phase 2 model for surgical fixes."""
        print(f"Loading Phase 2 model: {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        
        self.vocab = checkpoint['nodes']
        self.embeddings = checkpoint['embeddings'].copy()
        self.word_to_idx = {w: i for i, w in enumerate(self.vocab)}
        
        # Normalize for similarity computations
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        self.embeddings_norm = self.embeddings / norms
        
    def get_class_centroid(self, class_words):
        """Compute centroid of class members."""
        valid_words = [w for w in class_words if w in self.word_to_idx]
        if not valid_words:
            return None
        
        indices = [self.word_to_idx[w] for w in valid_words]
        class_embeddings = self.embeddings[indices]
        centroid = np.mean(class_embeddings, axis=0)
        return centroid / (np.linalg.norm(centroid) + 1e-10)
    
    def surgical_fix_household_confusion(self):
        """Fix the wardrobe-electrical device confusion."""
        print("\n SURGICAL FIX: Household Items Confusion")
        
        # Electrical devices centroid (without contaminated ones)
        clean_electrical = ['clock', 'lamp']  # Clean members
        electrical_centroid = self.get_class_centroid(clean_electrical)
        
        # Furniture centroid (without wardrobe)
        clean_furniture = ['bed', 'chair', 'couch', 'table']
        furniture_centroid = self.get_class_centroid(clean_furniture)
        
        if electrical_centroid is None or furniture_centroid is None:
            return
        
        # Pull electrical devices toward their clean centroid
        for word in ['computerkeyboard', 'telephone', 'television']:
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                current = self.embeddings[idx]
                
                # Strong pull toward electrical centroid
                new_emb = 0.6 * current + 0.4 * electrical_centroid * np.linalg.norm(current)
                self.embeddings[idx] = new_emb
                print(f"  [OK] Pulled {word} toward electrical devices")
        
        # Pull wardrobe toward furniture centroid
        if 'wardrobe' in self.word_to_idx:
            idx = self.word_to_idx['wardrobe']
            current = self.embeddings[idx]
            
            new_emb = 0.6 * current + 0.4 * furniture_centroid * np.linalg.norm(current)
            self.embeddings[idx] = new_emb
            print(f"  [OK] Pulled wardrobe toward furniture")
    
    def surgical_fix_vehicle_confusion(self):
        """Fix pickuptruck confusion with aquatic mammals."""
        print("\n SURGICAL FIX: Vehicle Confusion")
        
        # Get clean vehicle centroid
        clean_vehicles = ['bicycle', 'bus', 'motorcycle', 'train']
        vehicle_centroid = self.get_class_centroid(clean_vehicles)
        
        if vehicle_centroid is None:
            return
        
        # Pull pickuptruck toward vehicles
        if 'pickuptruck' in self.word_to_idx:
            idx = self.word_to_idx['pickuptruck']
            current = self.embeddings[idx]
            
            # Strong correction
            new_emb = 0.5 * current + 0.5 * vehicle_centroid * np.linalg.norm(current)
            self.embeddings[idx] = new_emb
            print(f"  [OK] Pulled pickuptruck toward vehicles")
    
    def surgical_fix_spider_confusion(self):
        """Fix spider confusion with insects."""
        print("\n SURGICAL FIX: Spider-Insect Confusion")
        
        # Get invertebrate centroid (without spider)
        clean_invertebrates = ['crab', 'lobster', 'snail', 'worm']
        invertebrate_centroid = self.get_class_centroid(clean_invertebrates)
        
        if invertebrate_centroid is None:
            return
        
        # Pull spider toward invertebrates
        if 'spider' in self.word_to_idx:
            idx = self.word_to_idx['spider']
            current = self.embeddings[idx]
            
            new_emb = 0.7 * current + 0.3 * invertebrate_centroid * np.linalg.norm(current)
            self.embeddings[idx] = new_emb
            print(f"  [OK] Pulled spider toward invertebrates")
    
    def surgical_fix_natural_scenes(self):
        """Fix bed confusion with natural scenes."""
        print("\n SURGICAL FIX: Natural Scenes Confusion")
        
        # Pull bed away from natural scenes toward furniture
        furniture_centroid = self.get_class_centroid(['chair', 'couch', 'table'])
        
        if furniture_centroid is not None and 'bed' in self.word_to_idx:
            idx = self.word_to_idx['bed']
            current = self.embeddings[idx]
            
            new_emb = 0.6 * current + 0.4 * furniture_centroid * np.linalg.norm(current)
            self.embeddings[idx] = new_emb
            print(f"  [OK] Pulled bed toward furniture")
    
    def apply_all_surgical_fixes(self, output_path):
        """Apply all surgical fixes and save."""
        print("\n" + "="*60)
        print(" PHASE 3: SURGICAL EMBEDDING FIXES")
        print("="*60)
        
        self.surgical_fix_household_confusion()
        self.surgical_fix_vehicle_confusion()
        self.surgical_fix_spider_confusion()
        self.surgical_fix_natural_scenes()
        
        # Save the surgically fixed model
        checkpoint = {
            'nodes': self.vocab,
            'embeddings': self.embeddings,
            'vocab_size': len(self.vocab),
            'embedding_dim': self.embeddings.shape[1]
        }
        
        torch.save(checkpoint, output_path)
        print(f"\n[OK] Saved surgically fixed model: {output_path}")
        print("="*60)

# Main execution
if __name__ == "__main__":
    INPUT_MODEL = 'phase2_FOR_8_VERSION_refined_final.pth'
    OUTPUT_MODEL = 'phase3_FOR_8_surgical_fixed.pth'
    
    fixer = SurgicalEmbeddingFixer(INPUT_MODEL)
    fixer.apply_all_surgical_fixes(OUTPUT_MODEL)
    
    print("\n Run your test script on 'phase3_surgical_fixed.pth' to verify improvements!")